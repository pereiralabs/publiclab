import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class UtilityMapper extends Mapper<Object, Text, Text, LongWritable>
{
  //Initialize key and value pair
  private Text utility = new Text();
  private LongWritable amount = new LongWritable(1);

  public void map(Object key, Text value, Context context) throws IOException, InterruptedException
  {
    //Mapper receives line by line of the text file
    String row = value.toString();

    //Check whether the text line contains data or its empty
    if (!row.isEmpty())
    {
      //Split the data based in <space> deliminator
      String[] rowValues = row.split(" ");

      //Get utility
      utility.set(rowValues[0]);

      //Get amount
      amount.set(Long.parseLong(rowValues[2]));

      //Returns utility as key and amount as value
      context.write(utility, amount);
    }
  }
}

