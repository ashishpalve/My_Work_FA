library(RecordLinkage)
library(sqldf)
library(igraph)

#data("RLdata500")
data("RLdata10000")

RLdata10000 = rbind(RLdata10000, RLdata10000[c("79", "79", "420"),])

mydata = RLdata10000

# rpairs <- RLBigDataLinkage(RLdata500, mydata, blockfld = c(1, 7),
#                            phonetic = c(1, 3))
Blocking = c("by", "bm", "bd")
Clustering = c("fname_c1", "fname_c2", "lname_c1", "lname_c2")
Threshold = 0.8

Generate_Clusters <- function(mydata, Blocking, Clustering, Threshold){
  mydata$Row_No = 1:nrow(mydata)
  exclude_vars = names(mydata)[!names(mydata) %in% c(Clustering, Blocking)]
  
  rpairs <- RLBigDataDedup(mydata, strcmp=TRUE, 
                           blockfld = Blocking, exclude = exclude_vars)
  e_weights = epiWeights(rpairs)
  
  pairs = getPairs(e_weights, min.weight = Threshold, max.weight = 1)
  
  pairs$series = sort(rep(1:(nrow(pairs)/3),3))
  pairs$id = trimws(pairs$id, which = "both")
  pairs = pairs[is.na(pairs$id) == FALSE & pairs$id != "",]
  
  multi_pairs = sqldf("select id, count(*) as ct from pairs group by id having count(*) > 1")
  
  series_to_consolidate = unique(pairs$series[pairs$id %in% multi_pairs$id])
  
  pure_clusters = pairs[!pairs$series %in% series_to_consolidate,]
  clusters_to_consolidate = pairs[pairs$series %in% series_to_consolidate,]
  
  temp = clusters_to_consolidate[c("id", "series")]
  temp = unique(temp)
  
  test = merge(temp, temp, by = "id", all = TRUE)
  rm(temp, clusters_to_consolidate, series_to_consolidate, multi_pairs)
  test = test[test$series.x != test$series.y,]
  
  test$cons = paste(test$series.x, test$series.y, sep = "-")
  
  m = lapply(test$cons, function(x){
    a = sort(unlist(strsplit(x, "-")))
    return(a)
  })
  
  rm(test)
  
  m = unique(m)
  
  #calculate an adjacency matrix using outer and intersect
  adj <- outer(m,m,Vectorize(function(x,y) length(intersect(x,y))))
  
  #using the igraph package, convert the matrix to a graph and extract the components
  cmp <- components(graph.adjacency(adj))
  
  #cmp$membership is the assignment of each node to a component:
  cmp$membership
  
  aggregated_series = tapply(m, cmp$membership, Reduce, f=union)
  rm(m, adj, cmp)
  
  old_series = unique(pure_clusters$series)
  series_mapping = data.frame(old_series)
  series_mapping$Cluster_No = 1:nrow(series_mapping)
  
  max_clus = max(series_mapping$Cluster_No)
  cluster_allot = max_clus
  if (length(aggregated_series) > 0) {
    for(i in 1:length(aggregated_series)){
      cluster_allot = cluster_allot + 1
      old_series = aggregated_series[[i]]
      Cluster_No = rep(cluster_allot, length(old_series))
      temp1 = data.frame(old_series, Cluster_No)
      series_mapping = rbind(series_mapping, temp1)
      rm(temp1, old_series, Cluster_No)
    }
    rm(aggregated_series)
  }
  
  pairs_final = merge(pairs, series_mapping, by.x = "series", by.y = "old_series", all.x = TRUE)
  pairs_final$series = NULL
  pairs_final$is_match = NULL
  pairs_final$Weight = NULL
  pairs_final = unique(pairs_final)

  pairs_final$Cluster_No = as.integer(pairs_final$Cluster_No)
  pairs_final$Row_No = as.integer(pairs_final$Row_No)
  mydata$Row_No = as.integer(mydata$Row_No)
  
  Non_Duplicate = mydata[!mydata$Row_No %in% pairs_final$Row_No,]
  Non_Duplicate$Cluster_No = nrow(mydata)
  pairs_final$id = NULL
  pairs_final$Class_Type = "C"
  Non_Duplicate$Class_Type = "NC"
  
  sapply(pairs_final, class)
  sapply(Non_Duplicate,class)
  
  cols_to_change = names(Non_Duplicate)[sapply(pairs_final, class) != sapply(Non_Duplicate, class)]

  pairs_final[cols_to_change] <- lapply(cols_to_change, function(x) {
    
    if (class(pairs_final[[x]]) == "factor") {
      pairs_final[[x]] = as.character(pairs_final[[x]])
    }
    match.fun(paste0("as.", class(Non_Duplicate[[x]])))(pairs_final[[x]])
  })
  
  Final_Data = rbind(pairs_final, Non_Duplicate)
  Final_Data = Final_Data[order(Final_Data$Cluster_No, Final_Data$Row_No),]
  return(Final_Data)
}

Result = Generate_Clusters(mydata, Blocking, Clustering, Threshold)
