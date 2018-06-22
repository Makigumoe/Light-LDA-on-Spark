#light lda on spark
利用spark的mapPartition进行实现。当全部节点计算完成后，shuffle并更新z，nkv，ndk，nk。
在单机版spark上，partition=20，k=10的条件下，spark lda的online模式训练耗时54s。light lda耗时118s。
这里肯定还有什么能优化的地方233333