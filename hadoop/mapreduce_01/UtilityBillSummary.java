import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class UtilityBillSummary {
  public static void main(String[] args) throws Exception {
          //Initiate Hadoop Configuration class
          Configuration conf = new Configuration();

          //Setting up job config
          Job job = Job.getInstance(conf, "UtilityBillSummary");
          job.setJarByClass(UtilityBillSummary.class);

          //Map and Reduce classes
          job.setMapperClass(UtilityMapper.class);
          job.setReducerClass(UtilityReducer.class);

          //Mapreduce key pairs
          job.setOutputKeyClass(Text.class);
          job.setOutputValueClass(LongWritable.class);

          //Assigning input and output files
          FileInputFormat.addInputPath(job, new Path(args[0]));
          FileOutputFormat.setOutputPath(job, new Path(args[1]));

          //Wait until job finishes
          System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}

