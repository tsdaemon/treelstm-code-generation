import edu.stanford.nlp.process.WordTokenFactory;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.util.StringUtils;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.parser.lexparser.TreeBinarizer;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.Trees;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.HashMap;
import java.util.Properties;
import java.util.Scanner;

public class ConstituencyParse {
  private BufferedWriter categoryWriter;
  private BufferedWriter parentWriter;
  private LexicalizedParser parser;
  private TreeBinarizer binarizer;
  private CollapseUnaryTransformer transformer;
  private GrammaticalStructureFactory gsf;

  private static final String PCFG_PATH = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";

  public ConstituencyParse(String parentPath, String categoriesPath) throws IOException {
    parentWriter = new BufferedWriter
        (new OutputStreamWriter(new FileOutputStream(parentPath), StandardCharsets.UTF_8));

    categoryWriter = new BufferedWriter
        (new OutputStreamWriter(new FileOutputStream(categoriesPath), StandardCharsets.UTF_8));

    parser = LexicalizedParser.loadModel(PCFG_PATH);
    binarizer = TreeBinarizer.simpleTreeBinarizer(
      parser.getTLPParams().headFinder(), parser.treebankLanguagePack());
    transformer = new CollapseUnaryTransformer();

    // set up to produce dependency representations from constituency trees
    TreebankLanguagePack tlp = new PennTreebankLanguagePack();
    gsf = tlp.grammaticalStructureFactory();
  }

  public List<HasWord> sentenceToTokens(String line) {
    List<HasWord> tokens = new ArrayList<>();
    for (String word : line.split(" ")) {
      tokens.add(new Word(word));
    }
    return tokens;
  }

  public Tree parse(List<HasWord> tokens) {
    Tree tree = parser.apply(tokens);
    return tree;
  }

  public int[] constTreeParents(Tree tree) {
    Tree binarized = binarizer.transformTree(tree);
    Tree collapsedUnary = transformer.transformTree(binarized);
    Trees.convertToCoreLabels(collapsedUnary);
    collapsedUnary.indexSpans();
    List<Tree> leaves = collapsedUnary.getLeaves();
    int size = collapsedUnary.size() - leaves.size();
    int[] parents = new int[size];
    HashMap<Integer, Integer> index = new HashMap<Integer, Integer>();

    int idx = leaves.size();
    int leafIdx = 0;
    for (Tree leaf : leaves) {
      Tree cur = leaf.parent(collapsedUnary); // go to preterminal
      int curIdx = leafIdx++;
      boolean done = false;
      while (!done) {
        Tree parent = cur.parent(collapsedUnary);
        if (parent == null) {
          parents[curIdx] = 0;
          break;
        }

        int parentIdx;
        int parentNumber = parent.nodeNumber(collapsedUnary);
        if (!index.containsKey(parentNumber)) {
          parentIdx = idx++;
          index.put(parentNumber, parentIdx);
        } else {
          parentIdx = index.get(parentNumber);
          done = true;
        }

        parents[curIdx] = parentIdx + 1;
        cur = parent;
        curIdx = parentIdx;
      }
    }

    return parents;
  }

  public String[] constTreeCategories(Tree tree) {
    Tree binarized = binarizer.transformTree(tree);
    Tree collapsedUnary = transformer.transformTree(binarized);
    Trees.convertToCoreLabels(collapsedUnary);
    collapsedUnary.indexSpans();
    List<Tree> leaves = collapsedUnary.getLeaves();
    int size = collapsedUnary.size() - leaves.size();
    String[] categories = new String[size];
    HashMap<Integer, Integer> index = new HashMap<Integer, Integer>();

    int idx = leaves.size();
    int leafIdx = 0;
    for (Tree leaf : leaves) {
      Tree cur = leaf.parent(collapsedUnary); // go to preterminal
      int curIdx = leafIdx++;
      boolean done = false;
      while (!done) {
        Tree parent = cur.parent(collapsedUnary);
        if (parent == null) {
          categories[curIdx] = "S";
          break;
        }

        int parentIdx;
        int parentNumber = parent.nodeNumber(collapsedUnary);
        if (!index.containsKey(parentNumber)) {
          parentIdx = idx++;
          index.put(parentNumber, parentIdx);
        } else {
          parentIdx = index.get(parentNumber);
          done = true;
        }

        categories[curIdx] = cur.label().toString();
        cur = parent;
        curIdx = parentIdx;
      }
    }

    return categories;
  }

  public void printParents(int[] parents) throws IOException {
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

  public void printCategories(String[] categories) throws IOException {
    StringBuilder sb = new StringBuilder();
    int size = categories.length;
    for (int i = 0; i < size - 1; i++) {
      sb.append(categories[i]);
      sb.append(' ');
    }
    sb.append(categories[size - 1]);
    sb.append('\n');
    categoryWriter.write(sb.toString());
  }

  public void printNewLine() throws IOException {
    parentWriter.write("\n");
    categoryWriter.write("\n");
  }

  public void close() throws IOException {
    parentWriter.close();
    categoryWriter.close();
  }

  public static void main(String[] args) throws Exception {
    Properties props = StringUtils.argsToProperties(args);
    if (!props.containsKey("parentpath") || !props.containsKey("catpath")) {
      System.err.println(
        "usage: java ConstituencyParse -parentpath <parentpath> -catpath <catpath>");
      System.exit(1);
    }

    String parentPath = props.getProperty("parentpath");
    String categoryPath = props.getProperty("catpath");
    ConstituencyParse processor = new ConstituencyParse(parentPath, categoryPath);

    Scanner stdin = new Scanner(System.in);
    int count = 0;
    long start = System.currentTimeMillis();
    while (stdin.hasNextLine()) {
      String line = stdin.nextLine();

      if(line != null && !line.trim().isEmpty()) {
        List<HasWord> tokens = processor.sentenceToTokens(line);
        Tree parse = processor.parse(tokens);

        // produce parent pointer representation
        int[] parents = processor.constTreeParents(parse);
        String[] categories = processor.constTreeCategories(parse);

        processor.printParents(parents);
        processor.printCategories(categories);
      } else {
        processor.printNewLine();
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
    processor.close();
  }
}
