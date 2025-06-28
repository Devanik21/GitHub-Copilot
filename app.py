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
import tempfile # New: For creating temporary files for linting
import os       # New: For file system operations (e.g., deleting temp files)
import subprocess # New: For running external commands like flake8

import google.generativeai as genai # New: For LLM integration
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
    code_hash: str = "" # New: Unique hash for the code chunk content

class CodeParser:
    """Advanced code parser using AST and tokenization"""
    
    def __init__(self):
        self.chunk_types = {
            'function': 'Function Definition',
            'class': 'Class Definition', 
            'import': 'Import Statement',
            'docstring': 'Documentation String', # New: For docstrings
            'comment': 'Comment Block',         # New: For comments
            'variable_assignment': 'Variable Assignment', # New: For variable assignments
            'if_block': 'If Statement Block', # New: For if-else blocks
            'for_loop': 'For Loop Block',     # New: For for loops
            'while_loop': 'While Loop Block', # New: For while loops
            'try_except_block': 'Try-Except Block' # New: For try-except blocks
        }
    
    def parse_python_code(self, code: str, file_path: str = "input.py") -> List[CodeChunk]:
        """Parse Python code into meaningful chunks using AST and tokenization."""
        chunks = []
        lines = code.split('\n')
        
        # Helper to create a chunk and calculate hash
        def _create_chunk(content, chunk_type, start_line, end_line, name, complexity=1.0):
            import hashlib # Import hashlib locally for this helper
            code_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            return CodeChunk(
                content=content,
                chunk_type=chunk_type,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                name=name,
                complexity_score=complexity,
                code_hash=code_hash
            )

        try:
            tree = ast.parse(code)
            
            # First pass: Major structural elements (functions, classes, imports)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
                    complexity = self._calculate_complexity(node)
                    chunk_type = 'function' if isinstance(node, ast.FunctionDef) else 'class'
                    
                    chunks.append(_create_chunk(
                        content=chunk_content,
                        chunk_type=chunk_type,
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        name=node.name,
                        complexity=complexity
                    ))
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    chunk_content = lines[node.lineno-1] if node.lineno <= len(lines) else ""
                    
                    chunks.append(_create_chunk(
                        content=chunk_content,
                        chunk_type='import',
                        start_line=node.lineno,
                        end_line=node.lineno,
                        name=f"import_{node.lineno}",
                        complexity=1.0
                    ))
            
            # Second pass: Docstrings, variable assignments, and control flow blocks
            for node in ast.walk(tree):
                # Docstrings
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        # Heuristic: docstring is usually the first string literal in the body
                        docstring_start_line = -1
                        docstring_end_line = -1
                        if hasattr(node, 'body') and node.body:
                            first_stmt = node.body[0]
                            if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant) and isinstance(first_stmt.value.value, str):
                                docstring_start_line = first_stmt.lineno
                                docstring_end_line = first_stmt.end_lineno if hasattr(first_stmt, 'end_lineno') else first_stmt.lineno
                        
                        if docstring_start_line != -1:
                            chunks.append(_create_chunk(
                                content=docstring,
                                chunk_type='docstring',
                                start_line=docstring_start_line,
                                end_line=docstring_end_line,
                                name=f"docstring_of_{getattr(node, 'name', 'module')}",
                                complexity=1.0
                            ))

                # Variable Assignments
                if isinstance(node, ast.Assign):
                    target_name = ""
                    if node.targets:
                        if isinstance(node.targets[0], ast.Name):
                            target_name = node.targets[0].id
                        elif isinstance(node.targets[0], ast.Tuple):
                            target_name = ", ".join([t.id for t in node.targets[0].elts if isinstance(t, ast.Name)])
                    
                    chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
                    chunks.append(_create_chunk(
                        content=chunk_content,
                        chunk_type='variable_assignment',
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        name=f"assign_{target_name or node.lineno}",
                        complexity=1.0
                    ))

                # Control Flow Blocks (if, for, while, try)
                if isinstance(node, ast.If):
                    chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
                    chunks.append(_create_chunk(
                        content=chunk_content,
                        chunk_type='if_block',
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        name=f"if_block_{node.lineno}",
                        complexity=self._calculate_complexity(node)
                    ))
                elif isinstance(node, ast.For):
                    chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
                    chunks.append(_create_chunk(
                        content=chunk_content,
                        chunk_type='for_loop',
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        name=f"for_loop_{node.lineno}",
                        complexity=self._calculate_complexity(node)
                    ))
                elif isinstance(node, ast.While):
                    chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
                    chunks.append(_create_chunk(
                        content=chunk_content,
                        chunk_type='while_loop',
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        name=f"while_loop_{node.lineno}",
                        complexity=self._calculate_complexity(node)
                    ))
                elif isinstance(node, ast.Try):
                    chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
                    chunks.append(_create_chunk(
                        content=chunk_content,
                        chunk_type='try_except_block',
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        name=f"try_block_{node.lineno}",
                        complexity=self._calculate_complexity(node)
                    ))
            
            # Third pass: Comments (using tokenize for robustness)
            comment_lines = []
            current_comment_block_content = []
            current_comment_start_line = -1
            last_comment_line = -1

            token_generator = tokenize.generate_tokens(io.StringIO(code).readline)
            for toktype, tokval, (srow, scol), (erow, ecol), line in token_generator:
                if toktype == tokenize.COMMENT:
                    if not current_comment_block_content: # Start of a new block
                        current_comment_start_line = srow
                    current_comment_block_content.append(tokval)
                    last_comment_line = erow # Update last line of current comment
                else:
                    if current_comment_block_content: # End of a comment block
                        comment_content = "\n".join(current_comment_block_content)
                        chunks.append(_create_chunk(
                            content=comment_content,
                            chunk_type='comment',
                            start_line=current_comment_start_line,
                            end_line=last_comment_line, # Corrected end_line
                            name=f"comment_{current_comment_start_line}",
                            complexity=1.0
                        ))
                        current_comment_block_content = []
                        current_comment_start_line = -1
                        last_comment_line = -1
            
            # Add any trailing comment block
            if current_comment_block_content:
                comment_content = "\n".join(current_comment_block_content)
                chunks.append(_create_chunk(
                    content=comment_content,
                    chunk_type='comment',
                    start_line=current_comment_start_line,
                    end_line=last_comment_line, # Corrected end_line
                    name=f"comment_{current_comment_start_line}",
                    complexity=1.0
                ))
        
        except SyntaxError as e:
            st.error(f"Syntax error in code: {e}")
            return []
        except Exception as e: # Catch any other unexpected errors during parsing
            st.error(f"An unexpected error occurred during parsing: {e}")
            return []
        
        # Sort chunks by start line for consistent display
        chunks.sort(key=lambda c: c.start_line)
        return chunks
    
    def _calculate_complexity(self, node) -> float:
        """Calculate cyclomatic complexity of AST node"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)): # Added AsyncFor, AsyncWith
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.comprehension, ast.GeneratorExp)): # Added comprehensions/generator expressions
                complexity += 1
        
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

class CodeLinter:
    """Lints Python code using flake8."""
    
    def __init__(self):
        pass

    def lint_code(self, code: str) -> List[Dict]:
        """Lints the given Python code using flake8 and returns a list of issues."""
        try:
            # Use a temporary file to lint the code
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py', encoding='utf-8') as tmp_file:
                tmp_file.write(code)
                tmp_file_path = tmp_file.name

            # Run flake8 on the temporary file
            # --isolated: ignore config files
            # --exit-zero: always exit with 0, even if errors are found
            # --format=json: output in JSON format
            # --max-line-length=120: common standard, or make configurable
            result = subprocess.run(
                ['flake8', '--isolated', '--exit-zero', '--format=json', '--max-line-length=120', tmp_file_path],
                capture_output=True,
                text=True,
                check=False # Don't raise CalledProcessError for non-zero exit codes
            )
            
            os.remove(tmp_file_path) # Clean up temporary file

            if result.stdout:
                # Check for the specific problematic output pattern
                if result.stdout.strip().startswith("json json"):
                    st.error(f"Flake8 produced unexpected output: '{result.stdout.strip()}'. This often indicates a problem with your flake8 installation or environment. Please ensure flake8 is correctly installed and accessible in your PATH. Stderr: {result.stderr}")
                    return []

                # flake8 outputs a JSON object where keys are filenames
                # We only lint one file, so we expect one key
                lint_output = json.loads(result.stdout)
                # The value for our temp file path will be a list of errors
                # flake8 paths might be normalized, so we check if our temp_file_path is a substring
                for key in lint_output:
                    if os.path.basename(tmp_file_path) in key or tmp_file_path in key:
                        return lint_output[key]
                return []
            return []
        except FileNotFoundError:
            st.warning("Flake8 not found. Please install it (`pip install flake8`) to enable linting.")
            return []
        except json.JSONDecodeError:
            st.error(f"Error decoding flake8 output: {result.stdout}. Stderr: {result.stderr}")
            return []
        except Exception as e:
            st.error(f"An error occurred during linting: {e}")
            return []

class CodeRAGSystem:
    """Complete RAG system for code analysis"""
    
    def __init__(self, api_key: str = None):
        self.parser = CodeParser()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase(self.embedding_generator.embedding_dim)
        self.linter = CodeLinter() # New: Initialize the linter
        self.chunks = []
        self.linting_results = [] # New: Store linting results
        self.llm_model = self._initialize_llm(api_key)

    def _initialize_llm(self, api_key: str):
        """Initializes the Gemini LLM if an API key is provided."""
        if not api_key:
            return None
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            return model
        except Exception as e:
            st.warning(f"Could not initialize Gemini model: {e}. AI generation features will be disabled.")
            return None
    
    def process_code(self, code: str, file_path: str = "input.py") -> Dict:
        """Process code through the complete RAG pipeline"""
        # Step 1: Parse code into chunks
        self.chunks = self.parser.parse_python_code(code, file_path)
        
        if not self.chunks:
            return {"error": "No code chunks could be parsed"}
        
        # Step 2: Generate embeddings
        self.chunks = self.embedding_generator.generate_embeddings(self.chunks)
        
        # Step 3: Add to vector database
        # Clear previous database content before adding new chunks
        self.vector_db = VectorDatabase(self.embedding_generator.embedding_dim) # Re-initialize DB
        self.vector_db.add_chunks(self.chunks)
        
        # Step 4: Lint code (New)
        self.linting_results = self.linter.lint_code(code)

        # Recalculate chunk types count as parser now has more types
        chunk_type_counts = {chunk_type: sum(1 for c in self.chunks if c.chunk_type == chunk_type) 
                             for chunk_type in self.parser.chunk_types.keys()}

        return {
            "total_chunks": len(self.chunks),
            "chunk_types": chunk_type_counts,
            "avg_complexity": np.mean([c.complexity_score for c in self.chunks]) if self.chunks else 0.0,
            "linting_issues_count": len(self.linting_results) # New: Count of linting issues
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
            "has_docstrings": any(c.chunk_type == 'docstring' for c in self.chunks), # New
            "has_comments": any(c.chunk_type == 'comment' for c in self.chunks),     # New
            "import_count": sum(1 for c in self.chunks if c.chunk_type == 'import'),
            "variable_assignment_count": sum(1 for c in self.chunks if c.chunk_type == 'variable_assignment'), # New
            "avg_function_length": np.mean([c.end_line - c.start_line + 1 
                                         for c in self.chunks if c.chunk_type == 'function']) if any(c.chunk_type == 'function' for c in self.chunks) else 0.0
        }
        return patterns
    
    def suggest_refactoring(self, chunk: CodeChunk) -> str:
        """Provides a refactoring suggestion using an LLM if available, otherwise falls back to rules."""
        if self.llm_model:
            prompt = f"""
            You are an expert code reviewer. Analyze the following Python code chunk and provide a concise refactoring suggestion.
            Focus on improving readability, maintainability, or performance.
            If the code is already good, say so.

            Code Chunk Name: {chunk.name}
            Type: {chunk.chunk_type}
            Complexity Score: {chunk.complexity_score:.1f}
            Lines: {chunk.end_line - chunk.start_line + 1}

            ```python
            {chunk.content}
            ```

            Suggestion:
            """
            return self._generate_llm_response(prompt)
        else:
            # Fallback to rule-based suggestions
            if chunk.chunk_type == 'function' and chunk.complexity_score > 5:
                return f"Function '{chunk.name}' has high complexity ({chunk.complexity_score:.1f}). Consider breaking it down into smaller, more focused functions. (Rule-based suggestion)"
            if chunk.chunk_type == 'function' and (chunk.end_line - chunk.start_line + 1) > 30:
                return f"Function '{chunk.name}' is quite long ({chunk.end_line - chunk.start_line + 1} lines). Breaking it into smaller parts can improve readability. (Rule-based suggestion)"
            return "No specific refactoring suggestions for this chunk. Provide an API key for AI-powered suggestions. (Rule-based suggestion)"

    def generate_code_snippet(self, prompt: str) -> str:
        """Generates code using an LLM if available, otherwise falls back to a simulated response."""
        if self.llm_model:
            # We can enhance this by providing retrieved context from the codebase
            # For now, we'll just use the prompt directly.
            llm_prompt = f"""
            You are a helpful code generation assistant.
            Generate a Python code snippet for the following request.
            Provide only the raw code, without any explanation before or after the code block.

            Request: "{prompt}"

            Python Code:
            """
            return self._generate_llm_response(llm_prompt)
        else:
            return f"// AI Code Generation is disabled. Please provide a Gemini API key.\n// Simulated response for: '{prompt}'\npass"

    def _generate_llm_response(self, prompt: str) -> str:
        """Helper function to call the LLM and handle errors."""
        if not self.llm_model:
            return "LLM not initialized. Please provide a valid API key in the sidebar."
        try:
            response = self.llm_model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred while communicating with the LLM: {e}"

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
    api_key = st.sidebar.text_input("Enter Gemini API Key (for AI features)", type="password")
    
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
    
        st.markdown("""
        - **Static Code Analysis (Linting)**: Identifying potential issues
        - **AI-Powered Generation/Refactoring**: Using an LLM for code assistance
        """)

    # Re-initialize the system if the API key changes
    # A simple way to handle this is to use the key in the cache key, but for this app,
    # we can just re-create it. A more robust app might manage state better.
    rag_system = CodeRAGSystem(api_key=api_key)
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Code Input", "🔍 Search & Retrieval", "📊 Code Analytics", "🛠️ Code Review & Assist", "🎯 System Demo"]) # Added tab4, renamed tab4 to tab5
    
    with tab1:
        st.header("Code Parsing and Chunking")
        
        # Sample code for demo
        sample_code = '''import numpy as np
import pandas as pd
from typing import List, Optional

class DataProcessor:
    """A class for processing and analyzing data"""
    
    # This is a top-level comment
    GLOBAL_CONSTANT = 100 # A global variable assignment
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> pd.DataFrame:
        """Load data from file"""
        try:
            # Attempt to read the CSV file
            self.data = pd.read_csv(self.data_path)
            if self.data.empty:
                print("Loaded empty data.")
            return self.data
        except FileNotFoundError:
            print(f"File {self.data_path} not found")
            return None
        except Exception as e: # Catch all other exceptions
            print(f"An unexpected error occurred: {e}")
            return None
    
    def clean_data(self, columns: List[str]) -> pd.DataFrame:
        """Clean data by removing null values"""
        if self.data is not None:
            cleaned_data = self.data.dropna(subset=columns)
            return cleaned_data
        return None

def calculate_statistics(data: pd.DataFrame) -> dict:
    """Calculate basic statistics for numerical columns"""
    # Iterate through numerical columns
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
                        
                        # Display processing results (updated to 4 columns)
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric("Total Chunks", results['total_chunks'])
                        with col_b:  
                            st.metric("Avg Complexity", f"{results['avg_complexity']:.1f}")
                        with col_c:
                            st.metric("Unique Chunk Types", len(results['chunk_types']))
                        with col_d: # New metric for linting issues
                            st.metric("Linting Issues", results['linting_issues_count'])
                        
                        # Show chunk breakdown
                        st.subheader("Chunk Analysis")
                        chunk_df = pd.DataFrame(list(results['chunk_types'].items()), 
                                              columns=['Type', 'Count'])
                        chunk_df = chunk_df[chunk_df['Count'] > 0].sort_values(by='Count', ascending=False) # Filter out types with 0 count
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
            
            if st.button("🔍 Search Code", type="primary"): # Changed to primary button
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
                                st.write(f"**Code Hash:** {result['chunk'].code_hash}") # Display code hash
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
                        st.write(f"- Has Docstrings: {'✅' if patterns['has_docstrings'] else '❌'}") # New
                        st.write(f"- Has Comments: {'✅' if patterns['has_comments'] else '❌'}")     # New
                        st.write(f"- Import Statements: {patterns['import_count']}")
                        st.write(f"- Variable Assignments: {patterns['variable_assignment_count']}") # New
                    
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
                        ,'content': [c.content for c in rag_system.chunks] # Added content for hover
                    })
                    
                    fig = px.scatter(
                        viz_df, x='x', y='y', 
                        color='cluster',
                        size='complexity',
                        hover_data=['name', 'type', 'complexity', 'content'], # Added content to hover
                        title="Code Chunks Clustered by Semantic Similarity"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4: # New tab for Code Review & Assist
        st.header("🛠️ Code Review & AI Assist")

        if not rag_system.chunks:
            st.warning("⚠️ Please process some code first in the 'Code Input' tab to use these features.")
        else:
            st.subheader("Linting Results")
            if rag_system.linting_results:
                st.info(f"Found {len(rag_system.linting_results)} linting issues.")
                for issue in rag_system.linting_results:
                    st.code(f"Line {issue['line_number']}:{issue['column_number']} - {issue['code']} {issue['text']}")
            else:
                st.success("🎉 No linting issues found! Your code looks clean.")
            
            st.markdown("---")

            st.subheader("Refactoring Suggestions")
            if rag_system.chunks:
                # Create a dictionary for selectbox options, mapping display string to CodeChunk object
                chunk_options = {f"{c.name} ({c.chunk_type}, Lines {c.start_line}-{c.end_line})": c for c in rag_system.chunks}
                selected_chunk_key = st.selectbox(
                    "Select a code chunk for refactoring suggestions:",
                    options=list(chunk_options.keys())
                )
                if selected_chunk_key:
                    selected_chunk = chunk_options[selected_chunk_key]
                    st.code(selected_chunk.content, language='python')
                    suggestion = rag_system.suggest_refactoring(selected_chunk)
                    st.info(suggestion)
            else:
                st.info("No chunks available for refactoring suggestions.")

            st.markdown("---")
            
            st.subheader("AI Code Generation")
            if rag_system.llm_model:
                st.markdown("*(This feature uses the Gemini API to generate code based on your prompt.)*")
            else:
                st.markdown("*(This feature is currently in **simulation mode**. Enter a Gemini API key in the sidebar to enable live AI code generation.)*")
            gen_prompt = st.text_area(
                "Enter a prompt for code generation:",
                value="function to add two numbers",
                height=100,
                help="Describe the code you want to generate (e.g., 'class for a simple counter', 'read csv file')"
            )
            if st.button("✨ Generate Code", type="primary"):
                if gen_prompt.strip():
                    with st.spinner("Generating code..."):
                        generated_code = rag_system.generate_code_snippet(gen_prompt)
                    st.code(generated_code, language='python')
                else:
                    st.warning("Please enter a prompt for code generation.")

    with tab5: # Renamed from tab4
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
        steps = [ # Updated steps for new features
            "Code Input", "AST Parsing & Chunking", "Linting", 
            "Embedding Generation", "Vector Storage", "Semantic Search", 
            "AI Suggestions & Generation", "Result Display"
        ]
        
        y_positions = list(range(len(steps)))
        
        for i, step in enumerate(steps):
            fig.add_trace(go.Scatter(
                x=[i], y=[0],
                mode='markers+text', # Changed marker color slightly for visual variety
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
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False), # Increased height for longer step names
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key features
        st.subheader("Advanced LLM Techniques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Core RAG Components:** # Updated descriptions
            - **Abstract Syntax Tree (AST) parsing** for structural code understanding
            - **Fine-grained chunking** (functions, classes, docstrings, variables, control flow)
            - **Semantic embeddings** using transformer models
            - **Vector similarity search** with FAISS indexing
            - **Contextual retrieval** for precise code matching
            """)
        
        with col2:
            st.markdown(""" # Updated descriptions
            **AI Enhancement Features:** 
            - **Complexity analysis** using cyclomatic complexity
            - **Code clustering** for organization insights  
            - **Pattern recognition** for code quality assessment
            - **Static code analysis (Linting)** for identifying issues
            - **Simulated code generation** for rapid prototyping
            - **Rule-based refactoring suggestions**
            - **Multi-modal search** supporting natural language queries
            """)
        
        # Performance metrics
        if rag_system.chunks:
            st.subheader("Current Session Statistics")
            
            col1, col2, col3, col4 = st.columns(4) # Added col4
            with col1:
                st.metric("Processed Chunks", len(rag_system.chunks))
            with col2:
                st.metric("Embedding Dimension", rag_system.embedding_generator.embedding_dim)
            with col3:
                st.metric("Vector DB Size", len(rag_system.vector_db.chunks))
            with col4: # New metric
                st.metric("Linting Issues", len(rag_system.linting_results))

if __name__ == "__main__":
    main()
