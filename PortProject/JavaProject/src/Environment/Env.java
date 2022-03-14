package Environment;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;
import java.util.Stack;

import org.json.JSONException;

import com.anylogic.engine.gui.ExperimentHost;
import com.anylogic.engine.gui.IExperimentHost;

import simplifiedSimulator.Container;
import simplifiedSimulator.CustomExperiment;
import simplifiedSimulator.Observation;
import simplifiedSimulator.QueyCrane;
import simplifiedSimulator.Result;
import simplifiedSimulator.Simulation;
import simplifiedSimulator.State;
import simplifiedSimulator.Task;
import simplifiedSimulator.TwoDPoint;

public class Env {

	public static final String spliter = System.getProperty("file.separator");
	public static String taskFileName = ".." + spliter + "Data" + spliter + "7Ships.txt";
	public static String outTaskFileName = ".." + spliter + "Data" + spliter + "outContainer.txt";
	public static int truckAmount = 80;
	public static boolean ifShowInfo = true;
	public static String truckDispatchPolicy = null; // ORDERED, PRTT, RANDOM,
	public static String yardCraneSchedulePolicy = "FCFS"; // ORDERED, FCFS, GREEDY, RANDOM,
	public static HashMap<Integer, Communicator> com = new HashMap<Integer, Communicator>(); // <port_number, socket>

	public static boolean fixRandom = false;
	public static int seed = 666666;

	public static boolean optimizeMode = false;
	public static boolean collectMode = false;

	public static int[] actionSequence;
	public static int pointer = 0;

	private DecimalFormat df = new DecimalFormat("0.00");
	// Singleton
	private static final Env env = new Env();

	private Env() {
	}

	public static Env getInstance() {
		return env;
	}

	public void runFastMode() {
		CustomExperiment c = new CustomExperiment(null);
		c.run();
	}

	public void runWithPresentation() {
		try {
			Simulation s = new Simulation();
			IExperimentHost host = new ExperimentHost(s);
			s.setup(host);
			host.launch();
		} catch (Exception e) {
			System.exit(-1);
		}
	}

	Heuristic heuristic = new Heuristic();

	public int getRandomAction(Object state, boolean fixRandom, Random R) {
		State info = (State) state;

		int action;

		if (optimizeMode) {
			action = actionSequence[pointer];
			pointer++;
		} else {
			action = fixRandom ? R.nextInt(info.shipAmount * 4) : new Random().nextInt(info.shipAmount * 4);
		}
		// int action = heuristic.chenHeuristic(info);

		if (collectMode) {
			actionSequence[pointer] = action;
			pointer++;
		}

		return action;
	}

	public ArrayList<Integer> executedActions = new ArrayList<Integer>();

	public int getAction(Object state, boolean fixRandom, Random R) throws JSONException, IOException {

		State info = (State) state;

		if (info.portNum == 0) {
			int action = new Random().nextInt(6);
			while (true) {
				if (action != 0) {
					return action;
				} else {
					action = new Random().nextInt(6);
				}
			}
		}

		boolean isDone = info.isDone;
		int portNum = info.portNum;
		double reward = 0;

		if (isDone) {
			reward = getFinalReward(info.queyCranes);
		}
		// System.out.println("Env Port: "+portNum+" Communicator Type:
		// "+com.get(portNum).type);
		com.get(portNum).sendEnvInfo(info, reward);
		int action;
		if (!isDone) {
			action = com.get(portNum).getAction();
			executedActions.add(action);
		} else {
			com.get(portNum).getAction();

			com.get(portNum).sendEndInfo(info);
			/*
			 * System.out.println("Executed Actions in Server: "); for (Integer a :
			 * executedActions) { System.out.print(a+" "); } System.out.println();
			 */
			action = -1;
		}
		return action;
	}

	public int getAction(Object state, Observation observation, boolean fixRandom, Random R)
			throws JSONException, IOException {

		State info = (State) state;

		if (info.portNum == 0) {
			int action = new Random().nextInt(6);
			while (true) {
				Stack<Container> pile = (observation.block.get(new TwoDPoint(observation.bay, action))).containerPiles
						.get(0);

				if (action != observation.stack && (pile.size() < 6 + 1)) {
					return action;
				} else {
					action = new Random().nextInt(6);
				}

			}
		}

		boolean isDone = info.isDone;
		int portNum = info.portNum;
		double reward = 0;

		if (isDone) {
			reward = getFinalReward(info.queyCranes);
		}
		// System.out.println("Env Port: "+portNum+" Communicator Type:
		// "+com.get(portNum).type);
		com.get(portNum).sendEnvInfo(info, reward);
		int action;
		if (!isDone) {
			action = com.get(portNum).getAction();
			executedActions.add(action);
		} else {
			com.get(portNum).getAction();

			com.get(portNum).sendEndInfo(info);
			/*
			 * System.out.println("Executed Actions in Server: "); for (Integer a :
			 * executedActions) { System.out.print(a+" "); } System.out.println();
			 */
			action = -1;
		}
		return action;
	}

	public void showComInfo() {
		Set<Integer> keys = com.keySet();
		System.out.print("Ports: ");
		for (Integer item : keys) {
			System.out.print(item + " ");
		}
		System.out.println();
	}

	public double getFinalReward(ArrayList<QueyCrane> queyCranes) {
		double totalWaitTime = 0;
		for (QueyCrane qc : queyCranes) {
			for (Task t : qc.completedTaskList) {
				totalWaitTime -= t.waitTime;
			}
		}
		return Double.parseDouble(df.format(totalWaitTime));
	}

	public void showResult(Object result) {
		Result r = (Result) result;
		double totalWaitTime = 0;
		System.out.println();
		for (QueyCrane qc : r.QCs) {
			System.out.println("--------" + qc.name + "----------");
			for (Task t : qc.completedTaskList) {
				totalWaitTime += t.waitTime;
				System.out.print(df.format(t.waitTime) + " ");
			}
			System.out.println();
		}
		System.out.println("Total Wait Time: " + df.format(totalWaitTime));
		accumulatedTotalTime += totalWaitTime;
	}

	private double accumulatedTotalTime = 0;

	public void testHeuristic() {
		int episode = 100;
		for (int i = 0; i < episode; i++) {
			runFastMode();
		}
		System.out.println("---------------------------------------------");
		System.out.println("Average Wait Time: " + accumulatedTotalTime / episode);
	}

	public static void main(String[] args) throws IOException, JSONException {

		System.out.println("Current Working Path: " + System.getProperty("user.dir"));

		String taskFileName = "F:\\PortProject\\Data\\5Ships.txt";
		File file = new File(taskFileName);
		System.out.println(file.isFile() && file.exists());

		/*
		 * System.out.println(file.getAbsoluteFile()); if(file.isFile() &&
		 * file.exists()) { System.out.println(); }
		 */
		/*
		 * if(file.isDirectory()){ String[]ss=file.list(); for(String s:ss){
		 * System.out.println(s); } }
		 */
		Env e = Env.getInstance();
		e.runFastMode();
		// e.testHeuristic();

		/*
		 * String os = System.getProperties().getProperty("os.name").split(" ")[0];
		 * System.out.println("Windows".equals(os));
		 * 
		 * for (int i = 0; i<100000; i++) {
		 * System.out.println("------------------episode "+i+"----------------------");
		 * long start = System.currentTimeMillis(); Env e = new Env(); e.runFastMode();
		 * long end = System.currentTimeMillis(); System.out.println((end - start) /
		 * 1000.0 + " seconds"); }
		 */

	}
}
