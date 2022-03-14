package Environment;

import java.io.IOException;

import simplifiedSimulator.CustomExperiment;

public class Executor implements Runnable{
	
	private int portNum;
	private Communicator com;
	private String type;
	public Executor(int portNum, String type) {
		this.portNum = portNum;
		this.type = type;
		com = new Communicator(portNum, type);
		Env.com.put(portNum, com);
	}
	
	public void test() {
		System.out.println("Executor: "+portNum+ " Communicator: "+com.portNum);
	}
	
	public void execute() {
		CustomExperiment c;
		synchronized(this) {
			c = new CustomExperiment(null);
			String[] args = new String[] {String.valueOf(portNum)};
			c.setCommandLineArguments_xjal(args);
		}
		c.run();
	}
	
	public void startServer() {
		Thread t = new Thread(this);
		t.start();
	}
	
	public void run() {
		//System.out.println(type+" "+Thread.currentThread().getName());
		try {
			if (!com.isConnected) {
				com.startServer();
			}
			execute();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		//System.out.println(type+" "+Thread.currentThread().getName()+" finished!");
	}
	
	
	public static void main(String[] args) throws IOException {
		int portNum = 10000;
		int episodeAmount = 1;
		Executor e = new Executor(portNum, "train");
		e.com.startServer();
		for (int i=0; i<episodeAmount; i++) {
			CustomExperiment c = new CustomExperiment(null);
			String[] cmdLineArgs = new String[] {String.valueOf(portNum)};
			c.setCommandLineArguments_xjal(cmdLineArgs);
			c.run();
		}
	}
}


