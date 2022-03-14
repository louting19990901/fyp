package Environment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashSet;

public class Init {
	
	public ArrayList<InputTask> tasks = new ArrayList<InputTask>();
	public Init() {
		
	}
	
	public void readFile(String filePath) {
		try {
            String encoding="GBK";
            File file=new File(filePath);
            if(file.isFile() && file.exists()){ 
                InputStreamReader read = new InputStreamReader(
                new FileInputStream(file),encoding);
                BufferedReader bufferedReader = new BufferedReader(read);
                String lineTxt = null;
                while((lineTxt = bufferedReader.readLine()) != null){
                    //System.out.println(lineTxt);
                	String[] line = lineTxt.split("	");
                	/*
                	for (String temp : line) {
                		System.out.print(temp+" ");
                	}
                	*/
                	String source = line[2];
                	String destination = line[3];
                	String taskType = line[9];
                	String targetQC = line[7];
                	
                	InputTask it = new InputTask(source, destination, taskType, targetQC);
                	tasks.add(it);
                }
                read.close();
		    }else{
		        System.out.println("Cannot find file");
		    }
	    } catch (Exception e) {
	        System.out.println("Error when reading file");
	        e.printStackTrace();
	    }
	}
	
	public String[] getFileNames(String path) {
		File dir = new File(path);
		File[] files = dir.listFiles();
		String[] names = new String[files.length];
		for (int i=0;i<files.length;i++) {
			names[i] = files[i].getName();
		}
		return names;
	}
	
	HashSet<String> QCs = new HashSet<String>();
	HashSet<String> berthes = new HashSet<String>();
	HashSet<String> sourceYards = new HashSet<String>();
	HashSet<String> desYards = new HashSet<String>();
	HashSet<String> loadQCs = new HashSet<String>();
	HashSet<String> unloadQCs = new HashSet<String>();
	HashSet<BerthQCPairs> bqPairs = new HashSet<BerthQCPairs>();

	public boolean feasibilityCheck() {
		for(InputTask it : tasks) {
			QCs.add(it.targetQC);
			
			if (it.taskType.equals("LOAD")) {
				sourceYards.add(it.source);
				berthes.add(it.destinaton);
				loadQCs.add(it.targetQC);
				bqPairs.add(new BerthQCPairs(it.destinaton, it.targetQC));
			} else {
				desYards.add(it.destinaton);
				berthes.add(it.source);
				unloadQCs.add(it.targetQC);
				bqPairs.add(new BerthQCPairs(it.source, it.targetQC));
			}
		}
		showCheckInfo();
		return true;
	}
	
	public void showCheckInfo() {
		System.out.println("\nQCs:");
		for (String qc : QCs) {
			System.out.print(qc+" ");
		}
		System.out.println("\nBerthes:");
		for (String berth : berthes) {
			System.out.print(berth+" ");
		}
		System.out.println("\nsourceYards:");
		for (String sourceYard : sourceYards) {
			System.out.print(sourceYard+" ");
		}
		System.out.println("\ndesYards:");
		for (String desYard : desYards) {
			System.out.print(desYard+" ");
		}
		System.out.println("\nloadQCs:");
		for (String loadQC : loadQCs) {
			System.out.print(loadQC+" ");
		}
		System.out.println("\nunloadQCs:");
		for (String unloadQC : unloadQCs) {
			System.out.print(unloadQC+" ");
		}
		System.out.println("\nBerth-QC:");
		for (BerthQCPairs bqPair : bqPairs) {
			System.out.print(bqPair.berth+"-"+bqPair.qc+" ");
		}
	}
	
	public static void main(String[] args) {
		Init init = new Init();
		String path = "..\\Data";
		String[] fileNames = init.getFileNames(path);
		/*
		for (String name : fileNames) {
			System.out.println(name);
		}
		*/
		
		String fileName = fileNames[7];
		String filePath = "..\\Data\\" + fileName;
		init.readFile(filePath);
		init.feasibilityCheck();
		
	}
}



