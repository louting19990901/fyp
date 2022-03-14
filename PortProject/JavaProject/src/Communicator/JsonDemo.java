package Communicator;

import java.util.ArrayList;

import org.json.JSONException;
import org.json.JSONObject;

public class JsonDemo {
	
	public static void main(String[] args) throws JSONException {
		
		ArrayList<Integer> al = new ArrayList<Integer>();
		al.add(1);
		al.add(2);
		al.add(3);
		
		int [] testList = new int[] {1,2,3,4,5} ;
		JSONObject json = new JSONObject();
		json.put("name", "Aaron");
		json.put("age", 20);
		json.put("isDone", false);
		json.put("list", testList);
		json.put("arrayList", al);
		
		
		System.out.println(json.getString("name"));
		System.out.println(json.getInt("age"));
		System.out.println(json.getBoolean("isDone"));

		System.out.println(json.toString());
	}
}
