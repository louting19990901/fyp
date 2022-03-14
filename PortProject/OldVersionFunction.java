public ArrayList<Container> readOutContainersMid(String filePath){
    ArrayList<Container> toBeOutContainers = new ArrayList<Container>();
    ArrayList<String> outContainerData = new ArrayList<String>();
    readFile(outTaskFileName, outContainerData);
    for(int i =0;i<outContainerData.size(); i++) {
        // get the first container's data: "id, source, destination, type, bay, stack, pile"
        String []containerInfo = outContainerData.get(i).split(",");
        int id = Integer.parseInt(containerInfo[0]);
        Yard source = yards.get(Integer.parseInt(containerInfo[1]));
        int type = Integer.parseInt(containerInfo[3]);
        int bay = Integer.parseInt(containerInfo[4]);
        int stack = Integer.parseInt(containerInfo[5]);
        int pile0or1 = Integer.parseInt(containerInfo[6]);
        boolean haveDoublePile = source.block.get(new TwoDPoint(bay, stack)).isDoublePile;
        // if is the 20 container
        if(type == 1){
            if(!haveDoublePile) {
                // if have only one pile, container and pile did not match, skip this data
                continue;
            }
            
            int layer = source.block.get(new TwoDPoint(bay, stack)).containerPiles.get(pile0or1).size();
            if(layer < source.layerLimit) {
                Container c = initNewOutContainer(20, bay, stack, layer, source);
                source.block.get(new TwoDPoint(bay, stack)).containerPiles.get(pile0or1).push(c);
                source.reservedAmount++;
                toBeOutContainers.add(c);
                
                // if have next container data
                if(i+1<outContainerData.size()) {
                    //System.out.println("have next data");
                    String []nextContainerInfo = outContainerData.get(i+1).split(",");
                    int id1 = Integer.parseInt(nextContainerInfo[0]);
                    if(id ==id1) { // same id							
                       i++;// deal with it this round instead of next round
                       // demonstate the validlity
                       Yard source1 = yards.get(Integer.parseInt(nextContainerInfo[1]));
                       int type1 = Integer.parseInt(nextContainerInfo[3]);
                       if(type1 != 1) {
                           continue; // if it is not a 20 container,next container invalid
                       }
                       int bay1 = Integer.parseInt(nextContainerInfo[4]);
                       int stack1 = Integer.parseInt(nextContainerInfo[5]);
                       int pile0or1_1 = Integer.parseInt(nextContainerInfo[6]);
                       boolean haveDoublePile1 = source1.block.get(new TwoDPoint(bay1, stack1)).isDoublePile;
                       if(!haveDoublePile1) {
                           continue; // next container invalid
                       }
                       int layer1 = source1.block.get(new TwoDPoint(bay1, stack1)).containerPiles.get(pile0or1_1).size();
                       if(layer1 < source1.layerLimit) {
                           Container c1 = initNewOutContainer(20, bay1, stack1, layer1, source1);
                           c.twin = c1;
                           c1.twin = c;
                           source1.block.get(new TwoDPoint(bay1, stack1)).containerPiles.get(pile0or1_1).push(c1);
                           //System.out.println("\n---twin container pushed to" + pile0or1_1 +  "---\n");
                           source1.reservedAmount++;
                           toBeOutContainers.add(c1);			
                       }	 					 
                    }						 
                }
            }			
            
        }else {
            // 40 container
            if(haveDoublePile) {
                // have double pile, container and pile did not match, skip this data
                continue;
            }
            int layer = source.block.get(new TwoDPoint(bay, stack)).containerPiles.get(0).size();
            if(layer < source.layerLimit) {
                Container c = initNewOutContainer(40, bay, stack, layer, source);							 
                source.block.get(new TwoDPoint(bay, stack)).containerPiles.get(0).push(c);
                source.reservedAmount++;
                toBeOutContainers.add(c);
            }		 		 
        }
    }	 	
    
    return toBeOutContainers;
}

// createOutTasks
while(it.hasNext()){
	Container c = it.next();
	if(it.hasNext()){
		Container c1 = it.next();
		if(c.twin == null){
			System.out.println("c.twin == null");
		}
		if (c.twin == c1){
			System.out.println("\n\n------double source task generated-----\n\n");
			Task t = new Task(c, c1, c.source, c1.source, null, null);
			c.task = t;
			c1.task =t;
			outs.add(t);
			outTaskNumbers++;
		}else{
			System.out.println("\n\n------single source task generated 2-----\n\n");
			Task t = new Task(c, null, c.source, null, null, null);
			Task t1 = new Task(c1, null, c1.source, null, null, null);
			c.task = t;
			c1.task = t1;
			outs.add(t);
			outs.add(t1);
			outTaskNumbers+=2;
		}
	}else{
		System.out.println("\n\n------single source task generated-----\n\n");
		Task t = new Task(c, null, c.source, null, null, null);
		c.task = t;
		outs.add(t);
		outTaskNumbers++;
	}
}


public ArrayList<Container> readOutContainers2(String filePath){
    ArrayList<Container> toBeOutContainers = new ArrayList<Container>();
    ArrayList<String> outContainerData = new ArrayList<String>();
    readFile(outTaskFileName, outContainerData);
    Iterator<String> containerIter = outContainerData.iterator();
    while(containerIter.hasNext()) {
        String []containerInfo = containerIter.next().split(",");
        int id = Integer.parseInt(containerInfo[0]);
        Yard source = yards.get(Integer.parseInt(containerInfo[1]));
        int type = Integer.parseInt(containerInfo[3]);
        int size = type==0?40:20;
        int bay = Integer.parseInt(containerInfo[4]);
        int stack = Integer.parseInt(containerInfo[5]);
        int pile0or1 = Integer.parseInt(containerInfo[6]);
        boolean haveDoublePile = source.block.get(new TwoDPoint(bay, stack)).isDoublePile;
        boolean validity = (type == 0 && !haveDoublePile && pile0or1 == 0) || (type == 1 && haveDoublePile);
        if(validity) {
            System.out.println("getting a container");
            Stack<Container> pile = source.block.get(new TwoDPoint(bay, stack)).containerPiles.get(pile0or1);
            int layer = pile.size();
            if(layer < source.layerLimit) {
                // generate first container
                Container c = initNewOutContainer(size, bay, stack, layer, pile0or1, source);							 
                pile.push(c);
                source.reservedAmount++;
                toBeOutContainers.add(c);
                // check if the 20 container has twin
                if(type == 1 && containerIter.hasNext()) {
                    //System.out.println("getting other twin container");
                    String []containerInfo1 = containerIter.next().split(",");
                    int id1 = Integer.parseInt(containerInfo1[0]);
                    Yard source1 = yards.get(Integer.parseInt(containerInfo1[1]));
                    int type1 = Integer.parseInt(containerInfo1[3]);
                    int bay1 = Integer.parseInt(containerInfo1[4]);
                    int stack1 = Integer.parseInt(containerInfo1[5]);
                    int pile0or1_1 = Integer.parseInt(containerInfo1[6]);
                    //System.out.println("twin pile:"+ pile0or1_1);
                    boolean haveDoublePile1 = source1.block.get(new TwoDPoint(bay1, stack1)).isDoublePile;
                    boolean validity1 = (id == id1) && (type1 == 1) && haveDoublePile1 && (pile0or1_1 == (pile0or1 == 0?1:0));
                    if(validity1) {
                        Stack<Container> pile1 = source1.block.get(new TwoDPoint(bay1, stack1)).containerPiles.get(pile0or1_1);
                        int layer1 = pile1.size();
                        if(layer1 < source1.layerLimit) {
                            Container c1 = initNewOutContainer(20, bay1, stack1, layer1, pile0or1_1, source1);
                            c.twin = c1;
                            c1.twin = c;
                            pile1.push(c1);
                            source1.reservedAmount++;
                            toBeOutContainers.add(c1);
                        }
                    }
                }
            }
        }
    }	 			 
    return toBeOutContainers;
}

// out container, currently in yard
public ArrayList<Container> readOutContainers1(String filePath){
    ArrayList<Container> toBeOutContainers = new ArrayList<Container>();
    ArrayList<String> outContainerData = new ArrayList<String>();
    boolean findTwinContainer = false;
    // read the file to get data and store in outContainerData
    readFile(outTaskFileName, outContainerData);
    // id, source, destination, type(0: 40, 1: 20), bay, stack, pile(0 or 1)
    for(String line: outContainerData) {
        //System.out.println(line);
     String []containerInfo = line.split(",");
     //System.out.println("\n\n" + containerInfo[0] + "\n\n");
     
     Yard source = yards.get(Integer.parseInt(containerInfo[1]));
     int type = Integer.parseInt(containerInfo[3]);
     System.out.println("type = " + type);
     int bay = Integer.parseInt(containerInfo[4]);
     int stack = Integer.parseInt(containerInfo[5]);
     
     if(type == 0) {
      
      int layer = source.block.get(new TwoDPoint(bay, stack)).containerPiles.get(0).size();
      //if is the container of size 40
      if ((!source.block.get(new TwoDPoint(bay, stack)).isDoublePile)&&(layer < source.layerLimit)){
       Container c = new Container(40, false, true, false);
       c.bay = bay;
       c.stack = stack;
       c.layer = layer;
       c.source = source;
       c.block = source.block;
       source.block.get(new TwoDPoint(bay, stack)).containerPiles.get(0).push(c);
       source.reservedAmount++;
       toBeOutContainers.add(c);
      
      }
      // if is the container of size 20
     }else {
      int pile0or1 = Integer.parseInt(containerInfo[6]);
      
      if (source.block.get(new TwoDPoint(bay, stack)).isDoublePile){
          Stack<Container> pile = source.block.get(new TwoDPoint(bay, stack)).containerPiles.get(pile0or1);
          if (!(pile.size() < source.layerLimit)) {
              break;
          }
       Container c1 = new Container(20, false, true, false);
       c1.pile0or1 = pile0or1;
       c1.bay = bay;
       c1.stack = stack;
       c1.layer = pile.size();
       c1.source = source;
       c1.block = source.block;
       if(!findTwinContainer) {
           System.out.println("find the first twin container");
           findTwinContainer = true;
           //System.out.println("twinContainer is null: " + twinContainer == null);
       }else {
           System.out.println("find the other twin container");
           Container preContainer = toBeOutContainers.get(toBeOutContainers.size()-1);
           preContainer.twin = c1;
           //System.out.println("twinContainer.twin is null: " + twinContainer.twin == null);
           c1.twin = preContainer;
           //System.out.println("c1.twin is null: " + c1.twin == null);
           findTwinContainer = false;
       }
       
       pile.push(c1);
       source.reservedAmount++;
       toBeOutContainers.add(c1);
       
      }
      
     }
     
    }  
    return toBeOutContainers;
   }
   


   public void GenerateContainerData(String fileName, int dataNumber, String sep, int []sourceRange, int [] destinationRange, int bayLimit, int []doublePileBayList, int stackLimit) {
    Random R = new Random();
 File file = new File(fileName);
 if (!file.exists()) {
     file.createNewFile();
 }
 FileWriter writer = new FileWriter(file.getAbsoluteFile());
 BufferedWriter buff = new BufferedWriter(writer);
 
 for(int i = 0; i< dataNumber; i++) {
     int index = i;
     int source = R.nextInt(sourceRange[1] + 1) + sourceRange[0];
     int destination = R.nextInt(sourceRange[1] + 1) + destinationRange[0]; 
     int type = R.nextInt(2);	        	
     int stack = R.nextInt(stackLimit);
     int pile = type == 0?0:R.nextInt(2);
     
     if(type == 0) {
         do {
             int bay = R.nextInt(bayLimit);
         }while()
         
         String data = index + sep + source + sep + destination + sep + type + sep + bay + sep + stack + sep + pile;
         
     }
     
     
 }
 
 
 
 
 
 

 for (int i = 0; i < dataNumber;i++){
     int pile;
     int type = R.nextInt(2);
     //int type = 0;
     if(type == 1){ // 双箱
         String data1 = (i+1) + sep+ R.nextInt(sourceLimit) + sep + sep + type + sep + R.nextInt(bayLimit) + sep + R.nextInt(stackLimit) + sep + R.nextInt(2) +"\n";
         String data2 = (i+1) + sep+ R.nextInt(sourceLimit) + sep + sep + type + sep + R.nextInt(bayLimit) + sep + R.nextInt(stackLimit) + sep + R.nextInt(2) +"\n";
         buff.write(data1);
         buff.write(data2);
     }else{
         //pile = type == 0?0:R.nextInt(2);
         String data = (i+1) + sep+ R.nextInt(sourceLimit) + sep + sep + type + sep + R.nextInt(bayLimit) + sep + R.nextInt(stackLimit) + sep + "0\n";
         buff.write(data);
     }
 }
//	        for(int i = 1;i<dataNum +1; i++){
//	            String data = (i+1) + sep+ R.nextInt(sourceLimit) + sep + sep + "0" + sep + R.nextInt(bayLimit) + sep + R.nextInt(stackLimit) + sep + "0\n";
//	            buff.write(data);
//	        }
 buff.close();
}

public boolean inArray()


