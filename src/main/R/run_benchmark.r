

source("./Tools.r")

suppressMessages(suppressWarnings(InstallAndLoadLibraries()))

dataPath = "../../../data"
dataList = list.files(dataPath,full.names = TRUE)
gridDictSize = 2:12

for(dataPath in dataList){
  
  absoluteDataPath = paste0(getwd(), "/",dataPath)
  datasetName = basename(dataPath)
  observedClass = unlist(read.csv(paste0(dataPath,"/",datasetName,"_label.csv")))
  
  ## Segmentation step
  print(absoluteDataPath)
  segmentation(absoluteDataPath)

  ## dictionary and categorical sequence clustering steps

  bestARI = -Inf
  for(dictSize in gridDictSize){
    categoricalSequences = dictionaryConstructionAndRecoding(absoluteDataPath, dictSize)
    ARI = categoricalSequenceClusteringAndARI(categoricalSequences, observedClass)
    print(ARI)
    bestARI = max(ARI, bestARI)
  }
  
 print(paste0("ARI on dataset ",datasetName,":", bestARI))
  
}

