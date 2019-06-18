#Check if required packages are present. Load if present else install and load
if(!require(data.table)){
  install.packages("data.table")
  library(data.table)
}
if(!require(openxlsx)){
  install.packages("openxlsx")
  library(openxlsx)
}
if(!require(sqldf)){
  install.packages("sqldf")
  library(sqldf)
}

library(data.table)
library(openxlsx)
library(sqldf)

setwd("/home/FRACTAL/ashish.palve/My_Work/Research/Buvana")
list.files()

load("RSAMPLE_jan_mar19.RData")
backlog = backlog_data_all_metrics_jan_mar19
actual = actual_data_all_metrics_jan_mar19
rm(actual_data_all_metrics_jan_mar19, backlog_data_all_metrics_jan_mar19)

backlog = backlog[backlog$quantity != 0 & is.na(backlog$quantity) == FALSE,]
actual = actual[actual$quantity != 0 & is.na(actual$quantity) == FALSE,]

# Reading input files
#backlog = read.xlsx("Investment Declaration  Form - FY 2019-20.xls.xlsx", sheet = "Backlog")
names(backlog)
backlog = backlog[names(backlog) %in% c("order_num", "order_line", "product_id", "status", "quantity", "refresh_date")]
backlog$refresh_date = as.Date(backlog$refresh_date, "%Y-%m-%d")

#actual = read.xlsx("Investment Declaration  Form - FY 2019-20.xls.xlsx", sheet = "Actual_file")
names(actual)
actual = actual[names(actual) %in% c("product_id", "order_num", "order_line", "billing_doc",
                                     "billing_item", "quantity", "refresh_date")]
actual$refresh_date = as.Date(actual$refresh_date, "%Y-%m-%d")

# Join the 2 files as a full join to avoid missing on data
# As the data is refreshed on same day for both tables, the transfer of items from backlog stage to actual stage will happen on the same day
# The join will give us chronological order of quantity movement
# We need to track the movement of goods across stages and check where the discrepancy takes place
joined = merge(backlog, actual, by = c("order_num", "order_line", "refresh_date"), all.x = TRUE, all.y = TRUE)
names(joined)[names(joined) == "quantity.x"] <- "quantity_x"
names(joined)[names(joined) == "quantity.y"] <- "quantity_y"

joined$quantity_x = ifelse(is.na(joined$quantity_x), 0, joined$quantity_x)
joined$quantity_y = ifelse(is.na(joined$quantity_y), 0, joined$quantity_y)
joined$billing_doc = ifelse(is.na(joined$billing_doc), 0, joined$billing_doc)
joined$billing_item = ifelse(is.na(joined$billing_item), 0, joined$billing_item)

# Compute the total quantity in both stages combined
joined$total_qty = joined$quantity_x + joined$quantity_y

joined$product_id.x = NULL
joined$product_id.y = NULL
joined$status = NULL
joined$document_type = NULL

# Format the dates
joined$refresh_date = as.Date(joined$refresh_date, origin = "1899-12-30")

# Order the data in chronological order of dates within each order number and order line group, to track the items week on week
joined = joined[order(joined$order_num, joined$order_line, joined$billing_doc, joined$billing_item, joined$refresh_date),]

joined = setDT(joined)

# Order and rank the events within each order number, order line and quantity_x(Backlog) and quantity_y(Actual)
# Order the data in ascending order of dates
# lot of times the quantity in backlog and actual is carried forwarded as it is and there is not changes in quantity in either backlog or actual
# Such non change weeks are of no use to us to calculate discrepancy as everything is unchanged
# If the backlog remains 200 for 3 weeks i.e. week1, week2 and week3 and after that 40 items are shipped to actual stage
# In such scenario within this group we rank the items in ascending by date
# This gives order/rank 1 to week1 which we will retain and drop all other weeks

# joined = joined[, order := rank(as.Date("1900-01-01")-refresh_date, ties.method = "first"), by = c("order_num", "order_line", "billing_doc", "billing_item",
#                                                                                                    "quantity_x", "quantity_y")]
joined = joined[, order := frank(refresh_date, ties.method = "first"), by = c("order_num", "order_line", "billing_doc", "billing_item",
                                                                              "quantity_x", "quantity_y")]
joined = as.data.frame(joined)

# Further in analysis you will see that if quantity stays same for a few week and after that there is no actual entry
# In such scenario with count 1 the discrepancy by above logic gets attributed to first week in that group
# We need to assign the discrepancy to the latest week after which there is no actual
# Hence we calculate the max value to be used later for such instances with count 1 and no change in quantity thereafter
max_date = aggregate(refresh_date ~ order_num + order_line,
                     data = joined, FUN = max)
names(max_date)[names(max_date) == "refresh_date"] <- "max_date"


joined1 = joined[joined$order == 1,]

joined1$billing_doc = NULL
joined1$billing_item = NULL
joined1$order = NULL

names(joined1)

# We are interested in total quantity dispatched in a block and not interested in individual block. Hence we will roll up the actual data
# But while rolling the actual data using above joined1 table, we double count the backlog quantity while aggregating the sum for actual
# This double counting happens because the backlog to actual join is a one to many join in many cases
# Hence we will split the backlog and actual values from above and take unique quantity for backlog while aggregate for actual
# After aggregating the actual we will join the backlog and actual tables again
# This step will give us the chronological order of quantity changes at order_num and order_line level
backlog_backup = joined1[c("order_num", "order_line", "refresh_date", "quantity_x")]
backlog_backup = unique(backlog_backup)
backlog_backup = backlog_backup[backlog_backup$quantity_x != 0,]

actual_backup = joined1[c("order_num", "order_line", "refresh_date", "quantity_y")]
actual_backup = actual_backup[actual_backup$quantity_y != 0,]
actual_backup = sqldf("select order_num, order_line, refresh_date, 
                sum(quantity_y) as quantity_y
                from actual_backup group by order_num, order_line, refresh_date")

joined2 = merge(backlog_backup, actual_backup, by = c("order_num", "order_line", "refresh_date"), all.x = TRUE, all.y = TRUE)
joined2$quantity_x = ifelse(is.na(joined2$quantity_x), 0, joined2$quantity_x)
joined2$quantity_y = ifelse(is.na(joined2$quantity_y), 0, joined2$quantity_y)

joined2$total_qty = joined2$quantity_x + joined2$quantity_y
joined2 = joined2[order(joined2$order_num, joined2$order_line, joined2$refresh_date),]

# This is just for testing purpose to access complex cases with many levels of backlog and actual dispatches
test = sqldf("select order_num, order_line, count(*) as ct from joined2 group by order_num, order_line having count(*) > 2")


# Now to calculate the gap we need lag values of backlog (quantity_x) and actual (quantity_y) values
joined2 = setDT(joined2)
joined2 = joined2[, lag_qty_x:=c(NA, quantity_x[-.N]), by=c("order_num", "order_line")]
joined2 = joined2[, lag_qty_y:=c(NA, quantity_y[-.N]), by=c("order_num", "order_line")]

# Computing the row count because in some examples there are 1 or more rows in backlog table with same quantity and then gets cancelled by customer
# There is no record in actual table for such cases. Such cases will only have 1 row in above table and thus no lag to be calculated
row_count = sqldf("select order_num, order_line, count(*) as ct from joined2 group by order_num, order_line")
joined2 = merge(joined2, row_count, by = c("order_num", "order_line"), all.x = TRUE)
joined2 = joined2[order(joined2$order_num, joined2$order_line, joined2$refresh_date),]

# If count is 1 and lag is NA it pertains to above case of order cancellation and hence is discrepancy
# If count is greater than 1 and lag quantity is NA it means it is the first record in a series of transaction in an order number and order line
# In such first record scenario we fill the NA in lag quantity with backlog quantity
joined2$lag_qty_x = ifelse(joined2$ct == 1 & is.na(joined2$lag_qty_x), NA,
                           ifelse(joined2$ct > 1 & is.na(joined2$lag_qty_x), joined2$quantity_x, joined2$lag_qty_x))
joined2$lag_qty_y = ifelse(joined2$ct == 1 & is.na(joined2$lag_qty_y), NA,
                           ifelse(joined2$ct > 1 & is.na(joined2$lag_qty_y), joined2$quantity_y, joined2$lag_qty_y))

# If count is 1 then discrepancy is backlog quantity minus actual quantity. This is for cancellation case where if order of 350 is cancelled then discrepancy is 350
# If count is greater than 1 and where lag quantity of backlog doesnt match the sum of backlog quantity and difference between actual quantity today and week before
joined2$discrepancy = ifelse(joined2$ct == 1 & joined2$quantity_x != joined2$quantity_y, 1,
                             ifelse((joined2$quantity_x + (joined2$quantity_y - joined2$lag_qty_y)) != joined2$lag_qty_x , 1, 0))
joined2$gap = ifelse(joined2$discrepancy == 1 & joined2$ct == 1, joined2$quantity_x-joined2$quantity_y,
                     ifelse(joined2$discrepancy == 1 & joined2$ct > 1, joined2$lag_qty_x + joined2$lag_qty_y - joined2$total_qty, 0))
joined2 = as.data.frame(joined2)

rm(backlog, backlog_backup, actual, actual_backup, joined1, row_count)

# Using the max date from earlier for cases where there was just an entry of backlog repeated as is over many weeks. 
# We need to assign the discrepancy to the last week till which no actual was found for that case
count1_items = joined2[joined2$ct == 1,]
noncount1_items = joined2[joined2$ct > 1,]

names(max_date)
count1_items = merge(count1_items, max_date, by = c("order_num", "order_line"), all.x = TRUE)
count1_items$refresh_date = count1_items$max_date
count1_items$max_date = NULL

joined3 = rbind(count1_items, noncount1_items)
joined3 = joined3[order(joined3$order_num, joined3$order_line, joined3$refresh_date),]

# Just exploring discrepancy cases for evaluation. Not part of delivery code
discrepancy_order  =sqldf("select order_num, order_line from joined3 where discrepancy = 1")
discrepancy_data = merge(discrepancy_order, joined3, by = c("order_num", "order_line"), all.x=TRUE)
discrepancy_data = discrepancy_data[discrepancy_data$ct > 1,]
discrepancy_data = discrepancy_data[order(discrepancy_data$order_num, discrepancy_data$order_line, discrepancy_data$refresh_date),]
sum(joined3$discrepancy, na.rm = TRUE)

# During the ordering stage and retaining last changed quantity row, we lost a lot of weeks from original data
# Hence get the order number, order line and refresh date from joined table
final = joined[c("order_num", "order_line", "refresh_date")]
final = unique(final)
final$refresh_date = as.Date(final$refresh_date,  origin = "1899-12-30")

final = merge(final, joined3[c("order_num", "order_line", "refresh_date", "gap", "discrepancy", "ct")],
              by = c("order_num", "order_line", "refresh_date"), all.x = TRUE)
final$gap = ifelse(is.na(final$gap), 0, final$gap)
final = final[order(final$order_num, final$order_line, final$refresh_date),]

# Write the final output
write.csv(final, "final_output.csv", row.names = FALSE)
