import edu.stanford.nlp.process.WordTokenFactory;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.nndep.DependencyParser;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;

public class DependencyParse {

  public static final String TAGGER_MODEL = "stanford-tagger/models/english-left3words-distsim.tagger";
  public static final String PARSER_MODEL = "edu/stanford/nlp/models/parser/nndep/english_SD.gz";

  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    if (!props.containsKey("parentpath") ||
        !props.containsKey("relpath")) {
      System.err.println(
        "usage: java DependencyParse -parentpath <parentpath> -relpath <relpath>");
      System.exit(1);
    }

    String parentPath = props.getProperty("parentpath");
    String relPath = props.getProperty("relpath");

    BufferedWriter parentWriter = new BufferedWriter
        (new OutputStreamWriter(new FileOutputStream(parentPath), StandardCharsets.UTF_8));

    BufferedWriter relWriter = new BufferedWriter
        (new OutputStreamWriter(new FileOutputStream(relPath), StandardCharsets.UTF_8));

    MaxentTagger tagger = new MaxentTagger(TAGGER_MODEL);
    DependencyParser parser = DependencyParser.loadFromModelFile(PARSER_MODEL);
    Scanner stdin = new Scanner(System.in);
    int count = 0;
    long start = System.currentTimeMillis();
    while (stdin.hasNextLine()) {
      String line = stdin.nextLine();
      if(line != null && !line.trim().isEmpty()) {
        List<HasWord> tokens = new ArrayList<>();
        for (String word : line.split(" ")) {
          tokens.add(new Word(word));
        }

        List<TaggedWord> tagged = tagger.tagSentence(tokens);

        int len = tagged.size();
        Collection<TypedDependency> tdl = parser.predict(tagged).typedDependencies();
        int[] parents = new int[len];
        for (int i = 0; i < len; i++) {
          // if a node has a parent of -1 at the end of parsing, then the node
          // has no parent.
          parents[i] = -1;
        }

        String[] relns = new String[len];
        for (TypedDependency td : tdl) {
          // let root have index 0
          int child = td.dep().index();
          int parent = td.gov().index();
          relns[child - 1] = td.reln().toString();
          parents[child - 1] = parent;
        }

        // print parent pointers
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len - 1; i++) {
          sb.append(parents[i]);
          sb.append(' ');
        }
        sb.append(parents[len - 1]);
        sb.append('\n');
        parentWriter.write(sb.toString());

        // print relations
        sb = new StringBuilder();
        for (int i = 0; i < len - 1; i++) {
          sb.append(relns[i]);
          sb.append(' ');
        }
        sb.append(relns[len - 1]);
        sb.append('\n');
        relWriter.write(sb.toString());
      } else {
        relWriter.write("\n");
        parentWriter.write("\n");
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
    parentWriter.close();
    relWriter.close();
  }
}
