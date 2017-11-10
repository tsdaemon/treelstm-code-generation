import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.util.StringUtils;
import uk.ac.ed.easyccg.main.EasyCCG;
import uk.ac.ed.easyccg.syntax.ParsePrinter;
import uk.ac.ed.easyccg.syntax.Parser;
import uk.ac.ed.easyccg.syntax.ParserAStar;
import uk.ac.ed.easyccg.syntax.TaggerEmbeddings;
import uk.ac.ed.easyccg.syntax.evaluation.Evaluate;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


import uk.ac.ed.easyccg.syntax.Category;
import uk.ac.ed.easyccg.syntax.InputReader;
import uk.ac.ed.easyccg.syntax.InputReader.InputToParser;
import uk.ac.ed.easyccg.syntax.ParsePrinter;
import uk.ac.ed.easyccg.syntax.Parser;
import uk.ac.ed.easyccg.syntax.ParserAStar;
import uk.ac.ed.easyccg.syntax.ParserAStar.SuperTaggingResults;
import uk.ac.ed.easyccg.syntax.SyntaxTreeNode;
import uk.ac.ed.easyccg.syntax.SyntaxTreeNode.SyntaxTreeNodeFactory;
import uk.ac.ed.easyccg.syntax.TagDict;
import uk.ac.ed.easyccg.syntax.TaggerEmbeddings;
import uk.ac.ed.easyccg.syntax.Util;
import uk.ac.ed.easyccg.syntax.evaluation.CCGBankDependencies;
import uk.ac.ed.easyccg.syntax.evaluation.CCGBankDependencies.DependencyParse;
import uk.ac.ed.easyccg.syntax.evaluation.Evaluate;
import uk.ac.ed.easyccg.syntax.evaluation.Evaluate.Results;
import uk.co.flamingpenguin.jewel.cli.ArgumentValidationException;
import uk.co.flamingpenguin.jewel.cli.CliFactory;
import uk.co.flamingpenguin.jewel.cli.Option;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Multimap;

public class CCGParse {

  private static int[] constTreeParents(SyntaxTreeNode tree) {
    List<SyntaxTreeNode.SyntaxTreeNodeLeaf> leaves = tree.getWords();
    int size = getSize(tree);

    int[] parents = new int[size];
    if(!tree.isLeaf()) {
      parseParents(parents, tree, 0, leaves.size() + 1);
    }

    return parents;
  }

  private static int parseParents(int[] parents, SyntaxTreeNode tree, int parentId, int currentId) {
    parents[currentId-1] = parentId;
    int nextId = currentId;
    for(SyntaxTreeNode node: tree.getChildren()) {
      if(node.isLeaf()) {
        parents[node.getHeadIndex()] = currentId;
      }
      else {
        nextId = parseParents(parents, node, currentId, nextId+1);
      }
    }
    return nextId;
  }

  private static int getSize(SyntaxTreeNode tree) {
    if (tree.isLeaf()) {
      return 1;
    } else {
      int sum = 1;
      for (SyntaxTreeNode child : tree.getChildren()) {
        sum += getSize(child);
      }
      return sum;
    }
  }

  private static void printParents(int[] parents, BufferedWriter parentWriter) throws IOException {
    StringBuilder sb = new StringBuilder();
    int size = parents.length;
    for (int i = 0; i < size - 1; i++) {
      sb.append(parents[i]);
      sb.append(' ');
    }
    sb.append(parents[size - 1]);
    sb.append('\n');
    parentWriter.write(sb.toString());
  }

  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    if (!props.containsKey("parentpath") || !props.containsKey("modelpath")) {
      System.err.println(
        "usage: java CCGParse -parentpath <parentpath> -modelpath <modelpath>");
      System.exit(1);
    }

    String parentPath = props.getProperty("parentpath");
    String modelPath = props.getProperty("modelpath");

    System.err.println("Loading model...");

    String[] rootCategories = new String[]{"S[dcl]", "S[wq]", "S[q]", "S[qem]", "NP"};

    Parser parser = new ParserAStar(
        new TaggerEmbeddings(new File(modelPath), 100, 0.0001,
            50),100,1,0.0,
        EasyCCG.InputFormat.TOKENIZED,
        Arrays.asList(rootCategories),
        new File(modelPath, "unaryRules"),
        new File(modelPath, "binaryRules"),
        new File(modelPath, "seenRules")
    );

    EasyCCG.OutputFormat outputFormat = EasyCCG.OutputFormat.CCGBANK;

    System.err.println("Parsing...");

    final ParserAStar.SuperTaggingResults supertaggingResults = new ParserAStar.SuperTaggingResults();

    BufferedWriter parentWriter = new BufferedWriter(new FileWriter(parentPath));

    Scanner stdin = new Scanner(System.in);
    int count = 0;
    long start = System.currentTimeMillis();
    while (stdin.hasNextLine()) {
      String line = stdin.nextLine();

      if(line != null && !line.trim().isEmpty()) {
        SyntaxTreeNode parse = parser.parse(supertaggingResults, line).get(0);

        // produce parent pointer representation
        int[] parents = constTreeParents(parse);

        printParents(parents, parentWriter);
      } else {
        parentWriter.write("\n");
      }

      count++;
      if (count % 1000 == 0) {
        double elapsed = (System.currentTimeMillis() - start) / 1000.0;
        System.err.printf("Parsed %d lines (%.2fs)\n", count, elapsed);
      }
    }

    parentWriter.close();

    long totalTimeMillis = System.currentTimeMillis() - start;
    System.err.printf("Done: %d lines in %.2fs (%.1fms per line)\n",
      count, totalTimeMillis / 1000.0, totalTimeMillis / (double) count);
  }
}
