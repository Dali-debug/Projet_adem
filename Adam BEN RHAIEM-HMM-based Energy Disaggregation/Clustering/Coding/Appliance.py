import numpy as np
import pandas as pd

class Appliance:
    def __init__(self,name,series,initial_means):
        initial_means.sort()
        self.name = name
        self.series = series
        self.clusteredSeries = series
        self.stateTransitions = []
        self.numberOfClusters = len(initial_means)
        self.transitionMatrix = [ [0 for j in range(len(initial_means))] for i in range(len(initial_means))]
        # Per Cluster
        self.means = initial_means
        self.numberOfPoints = [0 for i in range(len(initial_means))];
        self.mins = [np.inf for i in range(len(initial_means))];
        self.maxs = [-np.inf for i in range(len(initial_means))];
        self.covs = [ [0 for j in range(len(initial_means))] for i in range(len(initial_means))];
        self.seriesPerCluster = []
    
    ####### Single Cluster Methods    
    def findCluster(self,val):
        shortest_distance = abs(val - self.means[0]);
        closest_cluster = 0
        for i in range(1,self.numberOfClusters):
            current_distance = abs(val - self.means[i]);
            if(current_distance <= shortest_distance):
                shortest_distance = current_distance;
                closest_cluster = i
        return closest_cluster
            
    def calculateClusterMean(self,assignedCluster,val):
        self.means[assignedCluster] = (self.means[assignedCluster]*self.numberOfPoints[assignedCluster] + val)/(self.numberOfPoints[assignedCluster]+1)
        self.numberOfPoints[assignedCluster]+=1
    
    ####### All Clusters Methods
    def calculateClustersMeans(self):
        print("Mean Calculations Started.")
        rows_counter=0;
        for index, val in self.series.items():
            assignedCluster = self.findCluster(val)
            self.calculateClusterMean(assignedCluster,val)
            if(rows_counter%1000000==0):
                clear_output(wait=True)
                print(self)
            rows_counter+=1
        print("Mean Calculations Completed.")
        
    def classifyPoints(self):
        print("Point Classification Started.")
        for i in range(self.numberOfClusters):
            self.seriesPerCluster.append(self.series[list(map(lambda x : self.findCluster(x)==i, self.series))])
            print("Cluster "+str(i+1)+" determined.")
        print("Point Classification Finished.")
    
    def updateParameters(self):
        print("Parameters Update Started.")
        for i in range(self.numberOfClusters):
            self.mins[i] = np.min(self.seriesPerCluster[i])
            self.maxs[i] = np.max(self.seriesPerCluster[i])
            self.means[i] = np.mean(self.seriesPerCluster[i])
            self.covs[i] = [[np.var(self.seriesPerCluster[i])]]
            self.numberOfPoints[i] = len(self.seriesPerCluster[i])
            print("Cluster "+str(i+1)+" updated.")
        self.getClusteredSeries()
        self.getStateTransitions();
        self.calculateTransitionMatrix()
        print("Parameters Update Finished.")

        
    def updateClusters(self):
        self.calculateClustersMeans()
        clear_output(wait=True)
        self.classifyPoints()
        clear_output(wait=True)
        self.updateParameters()
        clear_output(wait=True)
        print(self)
        
    def getClusteredSeries(self):
        print("Creating Clustered Series.")
        self.clusteredSeries =  self.series.map(lambda x : self.findCluster(x))
        print("Clustered Series Created.")
            
    def getStateTransitions(self):
        i = 0
        temp = [];
        for index, val in self.clusteredSeries.items():
            if(i==0 or temp[i-1][0] != val):
                temp.append((val,1))
                i+=1
            else:
                temp[i-1] = (val,temp[i-1][1]+1)
            
        self.stateTransitions.append(temp[0])
        for i in range(1,len(temp)-1):
            ascending = temp[i][1]<=3 and (temp[i-1][0] < temp[i][0] and temp[i][0] < temp[i+1][0])
            descending = temp[i][1]<=3 and (temp[i-1][0] > temp[i][0] and temp[i][0] > temp[i+1][0])
#             print(temp[i-1],temp[i],temp[i+1],ascending,descending)
            if(not(ascending) and not(descending)):
                self.stateTransitions.append(temp[i])
#             else:
#                 self.stateTransitions[-1] = (self.stateTransitions[-1])
        self.stateTransitions.append(temp[-1]);
        print("getStateTransitions")
    
    def calculateTransitionMatrix(self):
        print("started calculating transition matrix")
        current_state = self.stateTransitions[0][0];
        print(self.transitionMatrix[current_state][current_state],self.stateTransitions[0][1])
        self.transitionMatrix[current_state][current_state]+= self.stateTransitions[0][1]-1
        for i in range(1,len(self.stateTransitions)):
            current_state = self.stateTransitions[i][0]
            previous_state = self.stateTransitions[i-1][0]
            self.transitionMatrix[current_state][current_state]+= self.stateTransitions[i][1]-1
            self.transitionMatrix[previous_state][current_state]+=1
        for i in range(self.numberOfClusters):
            temp_sum = sum(self.transitionMatrix[i])
            self.transitionMatrix[i] = list(map(lambda x : x/temp_sum, self.transitionMatrix[i]))
        print("finished calculating transition matrix")
        
    
    ####### Visualize
    def plotClusteredSeries(self):
        res = self.clusteredSeries.map(lambda x : self.means[x]);
        plt.figure(figsize=(10, 5))
        plt.plot(res.index, res, label="refrigerator",color="red")
        plt.xlabel('Time')
        plt.ylabel('Power (W)')
        plt.title('Power Consumption for refrigerator')
        plt.ylim(res.min() - 10, res.max() + 10)
        plt.legend()
        plt.show()
    
    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.series.index, self.series, label=self.name,color="red")
        plt.xlabel('Time')
        plt.ylabel('Power (W)')
        plt.title('Power Consumption for '+self.name)
        plt.ylim(self.series.min() - 10, self.series.max() + 10)
        colors = plt.cm.tab10(np.linspace(0, 1, self.numberOfClusters))
        for i in range(self.numberOfClusters): 
            plt.axhline(y=self.means[i], linestyle='--', label='_nolegend_',color=colors[i])
        plt.legend()
        plt.show()
        
    def __str__(self):
        
# a['ac']=np.array([[0.95,0.05],[0.05,0.95]])
# mean['ac']=np.array([[0],[1500]])
# cov['ac']=np.array([[[ 1.]],[[ 10]]])
        res = "======"+self.name+"======\n"
        res+="pi['"+self.name+"']=np.array("+str(list(map(lambda x : x/len(self.series),self.numberOfPoints)))+")\n"
        res+="a['"+self.name+"']=np.array("+str(self.transitionMatrix)+")\n"
        res+="mean['"+self.name+"']=np.array("+str(list(map(lambda x:[x],self.means)))+")\n"
        res+="cov['"+self.name+"']=np.array("+str(self.covs)+")\n"
        
        return res