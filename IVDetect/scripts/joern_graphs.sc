import gremlin.scala._
import org.apache.tinkerpop.gremlin.structure.Direction

import io.shiftleft.Implicits.JavaIteratorDeco
import io.shiftleft.codepropertygraph.generated._
import scala.io.Source
import scala.util.{Try,Success,Failure}
import java.nio.file.Paths

/* This file is APACHE licensed, based off of script from https://github.com/shiftleftsecurity/joern/ */

/** Some helper functions: adapted from ReachingDefPass.scala in codeproperty graph repo */
def vertexToStr(vertex: Vertex, identifiers: Map[Vertex,Int]): String = {
  val str = new StringBuffer()

  str.append("joern_id_")
  str.append("("+identifiers(vertex).toString + ")_")

  str.append("joern_code_")
  Try {
    val code1 = vertex.value2(NodeKeys.CODE).toString.replace("\'", "")
    val code2 = code1.replace("\"", "")
    val code = code2.replace("\\", "")
    str.append("("+ code +")"+ "_")
  }
  str.append("joern_type_")
  Try {
    str.append("("+vertex.label().toString+")"+ "_")
  }
  str.append("joern_name_")
  Try {
    str.append("("+vertex.value2(NodeKeys.NAME).toString+")"+ "_")
  }
  str.append("joern_line_")
  try {
    str.append("("+vertex.value2(NodeKeys.LINE_NUMBER).toString+")")
  }catch {
    case _: Throwable => str.delete(0, str.length())
    }
  

  str.toString
}

def toDot(graph: ScalaGraph): String = {
  var vertex_identifiers:Map[Vertex,Int] = Map()

  var index = 0
  graph.V.l.foreach{ v =>
    vertex_identifiers += (v -> index)
    index += 1
  }

  val buf = new StringBuffer()

  buf.append("digraph g {\n")

  buf.append("# AST\n")
  buf.append("{\n")
  buf.append("  edge[color=green3,constraint=true]\n")
  graph.E.hasLabel("AST").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# CFG\n")
  buf.append("{\n")
  buf.append("edge[color=red3,constraint=false]\n")
  graph.E.hasLabel("CFG").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# REF\n")
  buf.append("{\n")
  buf.append("edge[color=blue3,constraint=false]\n")
  graph.E.hasLabel("REF").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# CALL\n")
  buf.append("{\n")
  buf.append("edge[color=blue1,constraint=false]\n")
  graph.E.hasLabel("CALL").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# VTABLE\n")
  buf.append("{\n")
  buf.append("edge[color=yellow2,constraint=false]\n")
  graph.E.hasLabel("VTABLE").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# INHERITS_FROM\n")
  buf.append("{\n")
  buf.append("edge[color=violetred2,constraint=false]\n")
  graph.E.hasLabel("INHERITS_FROM").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# BINDS_TO\n")
  buf.append("{\n")
  buf.append("edge[color=grey37,constraint=false]\n")
  graph.E.hasLabel("BINDS_TO").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# REACHING_DEF\n")
  buf.append("{\n")
  buf.append("edge[color=grey37,constraint=false]\n")
  graph.E.hasLabel("REACHING_DEF").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# EVAL_TYPE\n")
  buf.append("{\n")
  buf.append("edge[color=grey37,constraint=false]\n")
  graph.E.hasLabel("EVAL_TYPE").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# CONTAINS\n")
  buf.append("{\n")
  buf.append("edge[color=grey37,constraint=false]\n")
  graph.E.hasLabel("CONTAINS").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# PROPAGATE\n")
  buf.append("{\n")
  buf.append("edge[color=grey37,constraint=false]\n")
  graph.E.hasLabel("PROPAGATE").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("# PDG\n")
  buf.append("{\n")
  buf.append("edge[color=grey37,constraint=false]\n")
  graph.E.hasLabel("CDG","REACHING_DEF").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    if ((childVertex.length() > 0)&&(parentVertex.length() > 0)){
      buf.append(s"""  "$parentVertex" -->> "$childVertex" \n """)
    }
  }
  buf.append("}\n")

  buf.append("}\n")

  buf.toString
}

@main def main(cpgFile: String): String = {
  loadCpg(cpgFile)
  toDot(cpg.graph)
}


/* read bug_names */