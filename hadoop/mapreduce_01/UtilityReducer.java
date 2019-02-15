import java.io.IOException;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class UtilityReducer extends Reducer<Text, LongWritable, Text, LongWritable>{
  private LongWritable result = new LongWritable();

  public void reduce(Text key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException{
    //Initiate variable total
    long total = 0;

    //Reduce key
    for (LongWritable val : values){
      total += val.get();
    }

    //Assign calculated value
    result.set(total);

    //Return key-value
    context.write(key, result);
  }
}

