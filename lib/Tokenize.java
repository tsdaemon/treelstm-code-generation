import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.WordTokenFactory;
import edu.stanford.nlp.util.StringUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;

public class Tokenize {
  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    if (!props.containsKey("tokpath")) {
      System.err.println(
        "usage: java Tokenize -tokpath <tokpath>");
      System.exit(1);
    }

    String tokPath = props.getProperty("tokpath");

    BufferedWriter tokWriter = new BufferedWriter
        (new OutputStreamWriter(new FileOutputStream(tokPath), StandardCharsets.UTF_8));

    Scanner stdin = new Scanner(System.in);
    int count = 0;
    long start = System.currentTimeMillis();
    while (stdin.hasNextLine()) {
      String line = stdin.nextLine();
      List<String> tokens = new ArrayList<>();
      PTBTokenizer<Word> tokenizer = new PTBTokenizer(
        new StringReader(line), new WordTokenFactory(), "");
      while(tokenizer.hasNext()) {
        tokens.add(tokenizer.next().word());
      }

      // print tokens
      int len = tokens.size();
      if(len > 0) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len - 1; i++) {
          sb.append(PTBTokenizer.ptbToken2Text(tokens.get(i)));
          sb.append(' ');
        }
        sb.append(PTBTokenizer.ptbToken2Text(tokens.get(len - 1)));
        sb.append('\n');
        tokWriter.write(sb.toString());
      } else {
        tokWriter.write("\n");
      }



      count++;
      if (count % 1000 == 0) {
        double elapsed = (System.currentTimeMillis() - start) / 1000.0;
        System.err.printf("Parsed %d lines (%.2fs)\n", count, elapsed);
      }
    }

    long totalTimeMillis = System.currentTimeMillis() - start;
    System.err.printf("Done: %d lines in %.2fs (%.1fms per line)\n",
      count, totalTimeMillis / 1000.0, totalTimeMillis / (double) count);
    tokWriter.close();
  }
}
