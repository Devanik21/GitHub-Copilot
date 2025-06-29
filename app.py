import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import ast
import tokenize
import io
import re
from typing import List, Dict, Tuple
import json
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time

# Configure page
st.set_page_config(
    page_title="RAG Code Analysis System",
    page_icon="🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class CodeChunk:
    """Represents a parsed code chunk with metadata"""
    content: str
    chunk_type: str  # function, class, docstring, import, etc.
    file_path: str
    start_line: int
    end_line: int
    name: str
    complexity_score: float
    embedding: np.ndarray = None

class CodeParser:
    """Advanced code parser using AST and tokenization"""
    
    def __init__(self):
        self.chunk_types = {
            'function': 'Function Definition',
            'class': 'Class Definition', 
            'import': 'Import Statement',
            'docstring': 'Documentation',
            'comment': 'Comment Block',
            'variable': 'Variable Assignment'
        }
    
    def parse_python_code(self, code: str, file_path: str = "input.py") -> List[CodeChunk]:
        """Parse Python code into meaningful chunks using AST"""
        chunks = []
        
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
                    complexity = self._calculate_complexity(node)
                    
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        chunk_type='function',
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        name=node.name,
                        complexity_score=complexity
                    ))
                
                elif isinstance(node, ast.ClassDef):
                    chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
                    complexity = self._calculate_complexity(node)
                    
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        chunk_type='class',
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        name=node.name,
                        complexity_score=complexity
                    ))
                
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    chunk_content = lines[node.lineno-1] if node.lineno <= len(lines) else ""
                    
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        chunk_type='import',
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=node.lineno,
                        name=f"import_{node.lineno}",
                        complexity_score=1.0
                    ))
        
        except SyntaxError as e:
            st.error(f"Syntax error in code: {e}")
            return []
        
        return chunks
    
    def _calculate_complexity(self, node) -> float:
        """Calculate cyclomatic complexity of AST node"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return float(complexity)

class EmbeddingGenerator:
    """Generate embeddings for code chunks using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Generate embeddings for all code chunks"""
        if not chunks:
            return chunks
        
        texts = [self._prepare_text_for_embedding(chunk) for chunk in chunks]
        
        with st.spinner("Generating embeddings..."):
            embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        return chunks
    
    def _prepare_text_for_embedding(self, chunk: CodeChunk) -> str:
        """Prepare code chunk text for embedding generation"""
        # Combine chunk type, name, and content for better semantic representation
        text_parts = [
            f"Type: {chunk.chunk_type}",
            f"Name: {chunk.name}",
            f"Content: {chunk.content}"
        ]
        return " ".join(text_parts)

class VectorDatabase:
    """FAISS-based vector database for similarity search"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.chunks = []
        self.is_built = False
    
    def add_chunks(self, chunks: List[CodeChunk]):
        """Add code chunks to the vector database"""
        if not chunks or not chunks[0].embedding is not None:
            return
        
        # Normalize embeddings for cosine similarity
        embeddings = np.array([chunk.embedding for chunk in chunks])
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        self.is_built = True
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Search for similar code chunks"""
        if not self.is_built:
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # Valid index
                results.append((self.chunks[idx], float(distances[0][i])))
        
        return results

class CodeRAGSystem:
    """Complete RAG system for code analysis"""
    
    def __init__(self):
        self.parser = CodeParser()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase(self.embedding_generator.embedding_dim)
        self.chunks = []
    
    def process_code(self, code: str, file_path: str = "input.py") -> Dict:
        """Process code through the complete RAG pipeline"""
        # Step 1: Parse code into chunks
        self.chunks = self.parser.parse_python_code(code, file_path)
        
        if not self.chunks:
            return {"error": "No code chunks could be parsed"}
        
        # Step 2: Generate embeddings
        self.chunks = self.embedding_generator.generate_embeddings(self.chunks)
        
        # Step 3: Add to vector database
        self.vector_db.add_chunks(self.chunks)
        
        return {
            "total_chunks": len(self.chunks),
            "chunk_types": {chunk_type: sum(1 for c in self.chunks if c.chunk_type == chunk_type) 
                          for chunk_type in set(c.chunk_type for c in self.chunks)},
            "avg_complexity": np.mean([c.complexity_score for c in self.chunks])
        }
    
    def search_code(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant code chunks based on query"""
        if not self.vector_db.is_built:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.model.encode([query])[0]
        
        # Search vector database
        results = self.vector_db.search(query_embedding, k)
        
        # Format results
        formatted_results = []
        for chunk, similarity in results:
            formatted_results.append({
                "chunk": chunk,
                "similarity": similarity,
                "type": chunk.chunk_type,
                "name": chunk.name,
                "complexity": chunk.complexity_score,
                "lines": f"{chunk.start_line}-{chunk.end_line}"
            })
        
        return formatted_results
    
    def generate_code_insights(self) -> Dict:
        """Generate insights about the codebase using LLM techniques"""
        if not self.chunks:
            return {}
        
        # Code complexity analysis
        complexities = [c.complexity_score for c in self.chunks]
        
        # Clustering analysis for code organization
        if len(self.chunks) > 1:
            embeddings = np.array([c.embedding for c in self.chunks])
            
            # Dimensionality reduction for visualization
            pca = PCA(n_components=min(2, embeddings.shape[1]))
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Clustering
            n_clusters = min(3, len(self.chunks))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            
            return {
                "complexity_stats": {
                    "mean": np.mean(complexities),
                    "std": np.std(complexities),
                    "max": np.max(complexities),
                    "min": np.min(complexities)
                },
                "clustering": {
                    "clusters": clusters.tolist(),
                    "reduced_embeddings": reduced_embeddings.tolist()
                },
                "code_patterns": self._analyze_code_patterns()
            }
        
        return {"complexity_stats": {"mean": np.mean(complexities)}}
    
    def _analyze_code_patterns(self) -> Dict:
        """Analyze common patterns in the codebase"""
        patterns = {
            "has_classes": any(c.chunk_type == 'class' for c in self.chunks),
            "has_functions": any(c.chunk_type == 'function' for c in self.chunks),
            "import_count": sum(1 for c in self.chunks if c.chunk_type == 'import'),
        }

        function_lengths = [c.end_line - c.start_line + 1
                            for c in self.chunks if c.chunk_type == 'function']
        
        patterns["avg_function_length"] = np.mean(function_lengths) if function_lengths else 0.0

        return patterns

# Initialize the RAG system
@st.cache_resource
def get_rag_system():
    return CodeRAGSystem()

def main():
    st.title("🦄 RAG Code Analysis System")
    st.markdown("**Advanced Code Understanding with Retrieval-Augmented Generation**")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    max_results = st.sidebar.slider("Max Search Results", 1, 10, 5)
    
    # LLM Techniques Information
    with st.sidebar.expander("🧠 LLM Techniques Used"):
        st.markdown("""
        **Advanced AI Techniques:**
        - **Semantic Embeddings**: Understanding code meaning
        - **Vector Similarity Search**: Finding relevant code
        - **AST Parsing**: Structural code analysis  
        - **Complexity Analysis**: Code quality metrics
        - **Clustering**: Code organization patterns
        - **Dimensionality Reduction**: Visualization
        - **Context-Aware Retrieval**: Precise code matching
        """)
    
    rag_system = get_rag_system()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Code Input", "🔍 Search & Retrieval", "📊 Code Analytics", "🎯 System Demo", "⚙️ How It Works"])
    
    with tab1:
        st.header("Code Parsing and Chunking")
        
        # Sample code for demo
        sample_code = '''import numpy as np
import pandas as pd
from typing import List, Optional

class DataProcessor:
    """A class for processing and analyzing data"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load data from file"""
        try:
            self.data = pd.read_csv(self.data_path)
            return self.data
        except FileNotFoundError:
            print(f"File {self.data_path} not found")
            return None
    
    def clean_data(self, columns: List[str]) -> pd.DataFrame:
        """Clean data by removing null values"""
        if self.data is not None:
            cleaned_data = self.data.dropna(subset=columns)
            return cleaned_data
        return None

def calculate_statistics(data: pd.DataFrame) -> dict:
    """Calculate basic statistics for numerical columns"""
    stats = {}
    for column in data.select_dtypes(include=[np.number]).columns:
        stats[column] = {
            'mean': data[column].mean(),
            'std': data[column].std(),
            'min': data[column].min(),
            'max': data[column].max()
        }
    return stats

def process_batch(file_paths: List[str]) -> Optional[pd.DataFrame]:
    """Process multiple files in batch"""
    all_data = []
    
    for path in file_paths:
        processor = DataProcessor(path)
        data = processor.load_data()
        
        if data is not None:
            cleaned = processor.clean_data(['id', 'value'])
            if cleaned is not None:
                all_data.append(cleaned)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None
'''
        
        code_input = st.text_area(
            "Enter Python Code:",
            value=sample_code,
            height=400,
            help="Paste your Python code here for analysis"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Process Code", type="primary"):
                if code_input.strip():
                    with st.spinner("Processing code through RAG pipeline..."):
                        results = rag_system.process_code(code_input)
                    
                    if "error" not in results:
                        st.success(f"✅ Successfully processed {results['total_chunks']} code chunks")
                        
                        # Display processing results
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Chunks", results['total_chunks'])
                        with col_b:  
                            st.metric("Avg Complexity", f"{results['avg_complexity']:.1f}")
                        with col_c:
                            st.metric("Chunk Types", len(results['chunk_types']))
                        
                        # Show chunk breakdown
                        st.subheader("Chunk Analysis")
                        chunk_df = pd.DataFrame(list(results['chunk_types'].items()), 
                                              columns=['Type', 'Count'])
                        st.bar_chart(chunk_df.set_index('Type'))
                        
                    else:
                        st.error(results['error'])
                else:
                    st.warning("Please enter some Python code to process")
    
    with tab2:
        st.header("Semantic Code Search")
        
        if not rag_system.vector_db.is_built:
            st.warning("⚠️ Please process some code first in the 'Code Input' tab")
        else:
            search_query = st.text_input(
                "Search Query:",
                placeholder="e.g., 'function that processes data', 'class for file handling', 'error handling code'",
                help="Enter a natural language description of what you're looking for"
            )
            
            if st.button("🔍 Search Code", type="primary"):
                if search_query.strip():
                    with st.spinner("Searching through code embeddings..."):
                        search_results = rag_system.search_code(search_query, max_results)
                    
                    if search_results:
                        st.success(f"Found {len(search_results)} relevant code chunks")
                        
                        for i, result in enumerate(search_results, 1):
                            with st.expander(f"Result {i}: {result['name']} (Similarity: {result['similarity']:.3f})"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.write(f"**Type:** {result['type']}")
                                with col2:
                                    st.write(f"**Lines:** {result['lines']}")
                                with col3:
                                    st.write(f"**Complexity:** {result['complexity']:.1f}")
                                
                                st.code(result['chunk'].content, language='python')
                    else:
                        st.info("No relevant code chunks found for your query")
                else:
                    st.warning("Please enter a search query")
    
    with tab3:
        st.header("Code Analytics & Insights")
        
        if not rag_system.chunks:
            st.warning("⚠️ Please process some code first to see analytics")
        else:
            insights = rag_system.generate_code_insights()
            
            if insights:
                # Complexity analysis
                if 'complexity_stats' in insights:
                    st.subheader("📈 Complexity Analysis")
                    stats = insights['complexity_stats']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Complexity", f"{stats['mean']:.2f}")
                    with col2:
                        st.metric("Max Complexity", f"{stats['max']:.0f}")
                    with col3:
                        st.metric("Min Complexity", f"{stats['min']:.0f}")
                    with col4:
                        st.metric("Std Deviation", f"{stats.get('std', 0):.2f}")
                
                # Code patterns
                if 'code_patterns' in insights:
                    st.subheader("🔍 Code Patterns")
                    patterns = insights['code_patterns']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Code Structure:**")
                        st.write(f"- Has Classes: {'✅' if patterns['has_classes'] else '❌'}")
                        st.write(f"- Has Functions: {'✅' if patterns['has_functions'] else '❌'}")
                        st.write(f"- Import Statements: {patterns['import_count']}")
                    
                    with col2:
                        st.write("**Quality Metrics:**")
                        avg_length = patterns.get('avg_function_length', 0)
                        st.write(f"- Avg Function Length: {avg_length:.1f} lines")
                        
                        if avg_length > 20:
                            st.write("⚠️ Consider breaking down long functions")
                        elif avg_length > 0:
                            st.write("✅ Good function length")
                
                # Clustering visualization
                if 'clustering' in insights:
                    st.subheader("🎯 Code Organization Clusters")
                    
                    clusters = insights['clustering']['clusters']
                    embeddings_2d = insights['clustering']['reduced_embeddings']
                    
                    # Create visualization dataframe
                    viz_df = pd.DataFrame({
                        'x': [e[0] for e in embeddings_2d],
                        'y': [e[1] for e in embeddings_2d],
                        'cluster': clusters,
                        'name': [c.name for c in rag_system.chunks],
                        'type': [c.chunk_type for c in rag_system.chunks],
                        'complexity': [c.complexity_score for c in rag_system.chunks]
                    })
                    
                    fig = px.scatter(
                        viz_df, x='x', y='y', 
                        color='cluster',
                        size='complexity',
                        hover_data=['name', 'type'],
                        title="Code Chunks Clustered by Semantic Similarity"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🎯 RAG System Architecture Demo")
        
        st.markdown("""
        This application demonstrates a complete **Retrieval-Augmented Generation (RAG)** system 
        specifically designed for code analysis, incorporating the same advanced techniques used in 
        AI coding assistants like GitHub Copilot.
        """)
        
        # Architecture diagram
        st.subheader("System Architecture")
        
        # Create a flow diagram
        fig = go.Figure()
        
        # Add boxes for each step
        steps = [
            "Code Input", "AST Parsing", "Chunk Extraction", 
            "Embedding Generation", "Vector Storage", "Semantic Search", "Result Generation"
        ]
        
        y_positions = list(range(len(steps)))
        
        for i, step in enumerate(steps):
            fig.add_trace(go.Scatter(
                x=[i], y=[0],
                mode='markers+text',
                marker=dict(size=50, color=f'rgba(55, 128, 191, {0.7 + i*0.05})'),
                text=step,
                textposition="middle center",
                name=step,
                showlegend=False
            ))
            
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=i+0.4, y=0,
                    ax=i+0.6, ay=0,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='rgb(55, 128, 191)'
                )
        
        fig.update_layout(
            title="RAG Pipeline Flow",
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key features
        st.subheader("Advanced LLM Techniques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Core RAG Components:**
            - **Abstract Syntax Tree (AST) parsing** for structural code understanding
            - **Semantic embeddings** using transformer models
            - **Vector similarity search** with FAISS indexing
            - **Contextual retrieval** for precise code matching
            """)
        
        with col2:
            st.markdown("""
            **AI Enhancement Features:**
            - **Complexity analysis** using cyclomatic complexity
            - **Code clustering** for organization insights  
            - **Pattern recognition** for code quality assessment
            - **Multi-modal search** supporting natural language queries
            """)
        
        # Performance metrics
        if rag_system.chunks:
            st.subheader("Current Session Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Processed Chunks", len(rag_system.chunks))
            with col2:
                st.metric("Embedding Dimension", rag_system.embedding_generator.embedding_dim)
            with col3:
                st.metric("Vector DB Size", len(rag_system.vector_db.chunks))

    with tab5:
        st.header("⚙️ How It Works: A Deep Dive into the RAG System")
        st.markdown("""
        This tab provides a detailed, step-by-step explanation of the technologies and processes used in this application. 
        Use this as a guide to understand how we turn raw code into a searchable, analyzable knowledge base.
        """)

        # Step 1: Parsing
        with st.expander("Step 1: Parsing & Chunking - Understanding Code Structure", expanded=True):
            st.markdown("""
            **Objective:** To break down a large, unstructured block of code into smaller, logical, and meaningful units called "chunks".

            - **What's Happening?**
                - We use Python's built-in `ast` (Abstract Syntax Tree) module. An AST is a tree representation of the code's structure, where each node represents a construct like a function definition, a class, or an import statement.
                - By "walking" this tree, we can precisely identify and extract these constructs, along with their metadata (e.g., name, start/end line numbers).
            
            - **Why is this important?**
                - **Precision:** Instead of blindly splitting a file by lines or blank spaces, AST parsing gives us semantically complete units. A function chunk contains the entire function body.
                - **Metadata:** We capture crucial context, like the name of the function (`node.name`) or its location in the file.
                - **Analysis:** This structured data allows us to perform further analysis, like calculating the **Cyclomatic Complexity** for each function or class to measure its potential complexity and maintainability.

            **In this app:** The `CodeParser` class is responsible for this step. When you click "Process Code", it's the first thing that runs.
            """)
            st.code("""
# Simplified view of AST parsing
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        # Extract function content, name, and line numbers
        ...
    elif isinstance(node, ast.ClassDef):
        # Extract class content, name, and line numbers
        ...
            """, language='python')

        # Step 2: Embedding
        with st.expander("Step 2: Embedding - Turning Code into Numbers", expanded=False):
            st.markdown("""
            **Objective:** To convert the text-based code chunks into numerical representations (vectors) that capture their semantic meaning.

            - **What's Happening?**
                - We use a pre-trained **Transformer model** from the `sentence-transformers` library (specifically, `all-MiniLM-L6-v2`).
                - This model has been trained on a massive amount of text and has learned to represent the meaning of sentences as high-dimensional vectors.
                - Before embedding, we enrich the code content with its metadata: `f"Type: {chunk.chunk_type} Name: {chunk.name} Content: {chunk.content}"`. This gives the model more context.

            - **Why is this important?**
                - **Semantic Understanding:** The resulting vectors (embeddings) place chunks with similar meanings close to each other in the vector space. For example, `def load_data()` and `def read_file()` would have similar embeddings, even though they use different words.
                - **Machine-Readable:** Computers can't compare text directly for meaning. But they are excellent at comparing numerical vectors. This step makes semantic search possible.

            **In this app:** The `EmbeddingGenerator` class handles this. It takes the list of `CodeChunk` objects and adds an `embedding` attribute to each one.
            """)

        # Step 3: Indexing
        with st.expander("Step 3: Indexing & Storage - Creating a Searchable Code Library", expanded=False):
            st.markdown("""
            **Objective:** To store the generated embeddings in a specialized database that allows for extremely fast similarity searches.

            - **What's Happening?**
                - We use `faiss` (Facebook AI Similarity Search), a high-performance library for vector search.
                - We create an `IndexFlatIP` index. "IP" stands for Inner Product, which is a mathematical operation to measure similarity between vectors. For normalized vectors (which we use), this is equivalent to **Cosine Similarity**.
                - All the embeddings from our code chunks are added to this index.

            - **Why is this important?**
                - **Speed:** Searching through thousands or millions of vectors one-by-one would be too slow. `faiss` uses optimized algorithms to find the "nearest neighbors" to a query vector almost instantly.
                - **Scalability:** This approach scales well to very large codebases.

            **In this app:** The `VectorDatabase` class wraps the `faiss` index. The `add_chunks` method populates the index.
            """)

        # Step 4: Retrieval
        with st.expander("Step 4: Retrieval & Search - Finding What You Need", expanded=False):
            st.markdown("""
            **Objective:** To find the most relevant code chunks based on a user's natural language query.

            - **What's Happening?**
                1.  The user's search query (e.g., "function that cleans data") is converted into an embedding using the *exact same* model from Step 2.
                2.  This new "query vector" is then used to search the `faiss` index.
                3.  `faiss` returns the top `k` most similar code chunk embeddings from the database, along with their similarity scores.

            - **Why is this important?**
                - This is the core of the **Retrieval** in RAG. It allows you to search based on *intent* and *meaning*, not just keywords. You don't need to remember the exact function name; you can describe what it does.

            **In this app:** The `search_code` method in `CodeRAGSystem` performs these steps. The results are then displayed in the "Search & Retrieval" tab.
            """)

        # Step 5: Analytics
        with st.expander("Bonus Step: Analytics & Insights - Going Beyond Search", expanded=False):
            st.markdown("""
            **Objective:** To leverage the embeddings and structured data for higher-level codebase analysis.

            - **What's Happening?**
                - **Clustering:** We use the `KMeans` algorithm on the code embeddings. This automatically groups semantically related functions and classes together. For example, all data loading and processing functions might end up in the same cluster.
                - **Visualization:** The embeddings are high-dimensional (384 dimensions for this model), which we can't visualize. We use **PCA (Principal Component Analysis)** to reduce them to 2 dimensions. This allows us to plot the chunks on a scatter plot, where proximity indicates semantic similarity.
                - **Pattern Analysis:** We can easily calculate metrics like the number of classes vs. functions, or the average function length, because we already parsed this data in Step 1.

            - **Why is this important?**
                - This provides a "bird's-eye view" of the codebase. It can help identify architectural patterns, find areas of related functionality, or spot code that might be overly complex or too long.

            **In this app:** The `generate_code_insights` method produces this data, which is then visualized in the "Code Analytics" tab.
            """)

if __name__ == "__main__":
    main()
