double euclid_dist_2(int    numCoords,
                    int    numObjs,
                    int    numClusters,
                    __global double *objects,     // [numCoords][numObjs]
                    __global double *clusters,    // [numCoords][numClusters]
                    int    objectId,
                    int    clusterId)
{
    int i;
    double ans=0.0;

    for (i = 0; i < numCoords; i++) {
        //ans += (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]) *
        //       (objects[numObjs * i + objectId] - clusters[numClusters * i + clusterId]);
        //printf("##%f %f\n",objects[objectId * numCoords + i] ,clusters[clusterId * numCoords + i]);
        ans += (objects[objectId * numCoords + i] - clusters[clusterId * numCoords + i]) *
               (objects[objectId * numCoords + i] - clusters[clusterId * numCoords + i]);
    }
    return ans;
}

__kernel void find_nearest_cluster(const int numClusters,
                                   const int numCoords,
                                   const int numObjs,
                                   __global double *objects,
                                   __global double *deviceClusters,
                                   __global int *membership)
{
    int objectId = get_global_id(0);
    int   index, i;
    double dist, min_dist;
    /*printf("###\n");
    if(objectId == 0){
        for(int i=0;i<numClusters;i++,printf("\n"))
            for(int j=0;j<numCoords;j++)
                printf("%f ",deviceClusters[i*numCoords+j]);
    }*/

    index    = 0;
    min_dist = euclid_dist_2(numCoords, numObjs, numClusters,
            objects, deviceClusters, objectId, 0);

    //printf("#%d %d %f\n",objectId,0,min_dist);
    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, numObjs, numClusters,
                objects, deviceClusters, objectId, i);
        //printf("#%d %d %f\n",objectId,i,dist);
        if (dist < min_dist) {
            min_dist = dist;
            index    = i;
        }
    }
	//printf("index %d\n",index);
    membership[objectId] = index;

    /*if (membership[objectId] != index) {
        membershipChanged[idx] = 1;
    }*/
}

