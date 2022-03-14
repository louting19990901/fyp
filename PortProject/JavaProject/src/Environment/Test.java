package Environment;

import java.util.ArrayList;
import java.util.Random;

public class Test {
	
	
	public void randomSeedTest() {
		Random r = new Random();
		r.setSeed(666);
        for (int j = 0; j < 5; j++) {
            System.out.print(" " + r.nextInt(100) + ", ");
        }
	
	}
	
	public void triOperator() {
		int a=2;
		int b=2;
		int c = a==b?3:4;
		System.out.println(c);
	}

	
	
	
	public static void main(String[] args) {
		Test t= new Test();
		t.triOperator();
	}

}
