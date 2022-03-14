import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class GenerateContainerData {
    public static void main(String[] args) throws IOException {
        int dataNum = 2500;
        String sep = ",";
        int sourceLimit = 90;
        int bayLimit = 25;
        int stackLimit = 6;
        String fileName = "outContainer.txt";

        Random R = new Random();
        File file = new File(fileName);
        if (!file.exists()) {
            file.createNewFile();
        }
        FileWriter writer = new FileWriter(file.getAbsoluteFile());
        BufferedWriter buff = new BufferedWriter(writer);

        for (int i = 0; i < dataNum;i++){
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
                String data = (i+1) + sep + R.nextInt(sourceLimit) + sep + sep + type + sep + R.nextInt(bayLimit) + sep + R.nextInt(stackLimit) + sep + "0\n";
                buff.write(data);
            }
        }
//        for(int i = 1;i<dataNum +1; i++){
//            String data = (i+1) + sep+ R.nextInt(sourceLimit) + sep + sep + "0" + sep + R.nextInt(bayLimit) + sep + R.nextInt(stackLimit) + sep + "0\n";
//            buff.write(data);
//        }
        buff.close();
    }
}

