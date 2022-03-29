package Communicator;

import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

import org.json.JSONException;
import org.json.JSONObject;

public class Demo implements Runnable {

	private ServerSocket server;
	private Socket client;
	private PrintWriter out;
	private InputStream in;

	// Singleton
	private static Demo demo;

	public static Demo getInstance() {
		if (demo == null) {
			demo = new Demo();
		}
		return demo;
	}
//	private void Demo() {}

	public void startThread() {
		Demo demo = getInstance();
		Thread t = new Thread(demo);
		t.start();
	}

	@Override
	public void run() {
		try {
			startServer();

			runModel();

			this.out.println("simulation finish");
			close();

		} catch (IOException | JSONException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void startServer() throws IOException {

		server = new ServerSocket(10000);
		client = server.accept();
		System.out.println(client.getInetAddress().getHostAddress() + " connected!");
		out = new PrintWriter(client.getOutputStream(), true);
		// in = new BufferedReader(new InputStreamReader(client.getInputStream()));
		in = client.getInputStream();

		/*
		 * for (int i=0;i<10;i++) { out.write(("Message from Server "+i).getBytes()); }
		 */

	}

	public void close() throws IOException {
		server.close();
		client.close();
	}

	public int getAction(int state, int reward, int currentStep) throws IOException, JSONException {

		boolean isDone;
		if (currentStep == totalStep - 1) {
			isDone = true;
		} else {
			isDone = false;
		}
		/*
		 * out.write(String.valueOf(state).getBytes());
		 * out.write(String.valueOf(reward).getBytes());
		 * out.write(String.valueOf(isDone).getBytes());
		 */

		// send json object
		JSONObject json = new JSONObject();
		json.put("state", state);
		json.put("reward", reward);
		json.put("isDone", isDone);
		out.println(json.toString());

		// get action
		// int action = Integer.parseInt(in.readLine());
		byte[] buffer = new byte[1024];
		int len = in.read(buffer);
		int action = Integer.parseInt(new String(buffer, 0, len));

		System.out.println("Server Received Action: " + action);

		return action;
	}

	public int totalStep = 10;
	int[] state = new int[] { 1, 5, 9, 8, 7, 3, 4, 6, 2, 0 };

	public void runModel() throws IOException, JSONException {

		for (int i = 0; i < totalStep; i++) {
			int reward;
			if (i == totalStep - 1) {
				reward = 10;
			} else {
				reward = 0;
			}
			int action = getAction(state[i], reward, i);
		}
	}

	public static void main(String[] args) {

		Demo demo = Demo.getInstance();
		Thread t = new Thread(demo);
		t.start();

	}

}
