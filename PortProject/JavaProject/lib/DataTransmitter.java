package Environment;

import java.util.ArrayList;

import simplifiedSimulator.State;

public class DataTransmitter {
	
	public boolean stateReady = false;
	public boolean actionReady = false;
	
	//public boolean flag = true;
	
	public State state;
	public int action;
	
	public DataTransmitter() {
		
	}
	
	public void setAction(int action) {
		this.action = action;
	}
	public void setState(Object state) {
		State info = (State)state;
		this.state = info;
	}
}
