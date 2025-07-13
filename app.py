import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import ast
import tokenize
import io
import re
from typing import List, Dict, Tuple, Optional, Set
import json
from dataclasses import dataclass, field
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import time
import hashlib
from collections import Counter, defaultdict
import warnings
import networkx as nx  # Add for call graph visualization
warnings.filterwarnings('ignore')

# Configure page with enhanced metadata
st.set_page_config(
    page_title="Advanced RAG Code Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Advanced RAG-powered Code Analysis with Semantic Understanding"
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .complexity-indicator {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .complexity-low { background-color: #d4edda; color: #155724; }
    .complexity-medium { background-color: #fff3cd; color: #856404; }
    .complexity-high { background-color: #f8d7da; color: #721c24; }
    .code-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class CodeChunk:
    """Enhanced code chunk with comprehensive metadata"""
    content: str
    chunk_type: str
    file_path: str
    start_line: int
    end_line: int
    name: str
    complexity_score: float
    embedding: Optional[np.ndarray] = None
    
    # Enhanced metadata
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    imports_used: Set[str] = field(default_factory=set)
    calls_made: Set[str] = field(default_factory=set)
    variables_defined: Set[str] = field(default_factory=set)
    
    # Quality metrics
    lines_of_code: int = 0
    comment_ratio: float = 0.0
    maintainability_index: float = 0.0
    
    def __post_init__(self):
        self.lines_of_code = len(self.content.split('\n'))
        self.comment_ratio = self._calculate_comment_ratio()
        self.maintainability_index = self._calculate_maintainability_index()
    
    def _calculate_comment_ratio(self) -> float:
        """Calculate ratio of comment lines to total lines"""
        lines = self.content.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        return comment_lines / max(len(lines), 1)
    
    def _calculate_maintainability_index(self) -> float:
        """Calculate maintainability index based on multiple factors"""
        # Simplified maintainability index calculation
        base_score = 100
        
        # Penalize high complexity
        complexity_penalty = min(self.complexity_score * 2, 20)
        
        # Reward good commenting
        comment_bonus = self.comment_ratio * 10
        
        # Penalize very long functions
        length_penalty = max(0, (self.lines_of_code - 20) * 0.5)
        
        return max(0, base_score - complexity_penalty + comment_bonus - length_penalty)
    
    @property
    def complexity_category(self) -> str:
        """Categorize complexity level"""
        if self.complexity_score <= 5:
            return "Low"
        elif self.complexity_score <= 10:
            return "Medium"
        elif self.complexity_score <= 20:
            return "High"
        else:
            return "Very High"
    
    @property
    def quality_score(self) -> float:
        """Overall quality score based on multiple metrics"""
        return (self.maintainability_index / 100) * 0.6 + \
               (min(self.comment_ratio * 10, 1)) * 0.2 + \
               (max(0, 1 - self.complexity_score / 20)) * 0.2

class AdvancedCodeParser:
    """Enhanced code parser with comprehensive AST analysis"""
    
    def __init__(self):
        self.chunk_types = {
            'function': 'Function Definition',
            'class': 'Class Definition', 
            'import': 'Import Statement',
            'docstring': 'Documentation',
            'comment': 'Comment Block',
            'variable': 'Variable Assignment',
            'constant': 'Constant Definition',
            'decorator': 'Decorator',
            'async_function': 'Async Function',
            'property': 'Property Definition'
        }
        
        # Track relationships between code elements
        self.call_graph = defaultdict(set)
        self.import_graph = defaultdict(set)
        self.inheritance_graph = defaultdict(set)
    
    def parse_python_code(self, code: str, file_path: str = "input.py") -> List[CodeChunk]:
        """Enhanced Python code parsing with comprehensive analysis"""
        chunks = []
        
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            
            # First pass: collect all imports and global context
            imports = self._extract_imports(tree, lines)
            chunks.extend(imports)
            
            # Second pass: analyze functions and classes with context
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    chunk = self._parse_function(node, lines, file_path)
                    if chunk:
                        chunks.append(chunk)
                
                elif isinstance(node, ast.AsyncFunctionDef):
                    chunk = self._parse_async_function(node, lines, file_path)
                    if chunk:
                        chunks.append(chunk)
                
                elif isinstance(node, ast.ClassDef):
                    chunk = self._parse_class(node, lines, file_path)
                    if chunk:
                        chunks.append(chunk)
                    
                    # Also parse methods within the class
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_chunk = self._parse_method(item, node, lines, file_path)
                            if method_chunk:
                                chunks.append(method_chunk)
            
            # Third pass: analyze module-level variables and constants
            module_vars = self._extract_module_variables(tree, lines, file_path)
            chunks.extend(module_vars)
            
        except SyntaxError as e:
            st.error(f"🚨 Syntax Error: {e}")
            st.info("💡 Try fixing the syntax errors in your code before analysis")
            return []
        except Exception as e:
            st.error(f"🚨 Parsing Error: {e}")
            return []
        
        return chunks
    
    def _parse_function(self, node: ast.FunctionDef, lines: List[str], file_path: str) -> Optional[CodeChunk]:
        """Parse function with comprehensive metadata extraction"""
        chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
        
        # Extract function metadata
        parameters = [arg.arg for arg in node.args.args]
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        docstring = ast.get_docstring(node)
        
        # Analyze function calls and variable usage
        calls_made = self._extract_function_calls(node)
        variables_defined = self._extract_variables(node)
        
        # Calculate various complexity metrics
        complexity = self._calculate_enhanced_complexity(node)
        
        return CodeChunk(
            content=chunk_content,
            chunk_type='async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function',
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.end_lineno,
            name=node.name,
            complexity_score=complexity,
            parameters=parameters,
            docstring=docstring,
            decorators=decorators,
            calls_made=calls_made,
            variables_defined=variables_defined
        )
    
    def _parse_async_function(self, node: ast.AsyncFunctionDef, lines: List[str], file_path: str) -> Optional[CodeChunk]:
        """Parse async function with async-specific analysis"""
        chunk = self._parse_function(node, lines, file_path)
        if chunk:
            chunk.chunk_type = 'async_function'
        return chunk
    
    def _parse_class(self, node: ast.ClassDef, lines: List[str], file_path: str) -> Optional[CodeChunk]:
        """Parse class with inheritance and method analysis"""
        chunk_content = '\n'.join(lines[node.lineno-1:node.end_lineno])
        
        # Extract class metadata
        decorators = [self._get_decorator_name(dec) for dec in node.decorator_list]
        docstring = ast.get_docstring(node)
        base_classes = [self._get_name(base) for base in node.bases]
        
        # Analyze class methods and properties
        methods = []
        properties = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
            elif isinstance(item, ast.FunctionDef) and any(
                isinstance(dec, ast.Name) and dec.id == 'property' 
                for dec in item.decorator_list
            ):
                properties.append(item.name)
        
        complexity = self._calculate_enhanced_complexity(node)
        
        chunk = CodeChunk(
            content=chunk_content,
            chunk_type='class',
            file_path=file_path,
            start_line=node.lineno,
            end_line=node.end_lineno,
            name=node.name,
            complexity_score=complexity,
            docstring=docstring,
            decorators=decorators
        )
        
        # Store inheritance information
        for base in base_classes:
            self.inheritance_graph[node.name].add(base)
        
        return chunk
    
    def _parse_method(self, node: ast.FunctionDef, class_node: ast.ClassDef, 
                     lines: List[str], file_path: str) -> Optional[CodeChunk]:
        """Parse class method with class context"""
        chunk = self._parse_function(node, lines, file_path)
        if chunk:
            chunk.name = f"{class_node.name}.{node.name}"
            chunk.chunk_type = 'method'
        return chunk
    
    def _extract_imports(self, tree: ast.AST, lines: List[str]) -> List[CodeChunk]:
        """Extract and analyze import statements"""
        import_chunks = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                chunk_content = lines[node.lineno-1] if node.lineno <= len(lines) else ""
                
                # Determine import type and modules
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                    import_type = "standard"
                else:  # ImportFrom
                    modules = [f"{node.module}.{alias.name}" for alias in node.names if node.module]
                    import_type = "from_import"
                
                import_chunks.append(CodeChunk(
                    content=chunk_content,
                    chunk_type='import',
                    file_path="input.py",
                    start_line=node.lineno,
                    end_line=node.lineno,
                    name=f"import_{node.lineno}_{import_type}",
                    complexity_score=1.0,
                    imports_used=set(modules)
                ))
        
        return import_chunks
    
    def _extract_module_variables(self, tree: ast.AST, lines: List[str], file_path: str) -> List[CodeChunk]:
        """Extract module-level variables and constants"""
        var_chunks = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                
                # Check if it's a constant (uppercase name)
                is_constant = var_name.isupper()
                chunk_type = 'constant' if is_constant else 'variable'
                
                chunk_content = lines[node.lineno-1] if node.lineno <= len(lines) else ""
                
                var_chunks.append(CodeChunk(
                    content=chunk_content,
                    chunk_type=chunk_type,
                    file_path=file_path,
                    start_line=node.lineno,
                    end_line=node.lineno,
                    name=var_name,
                    complexity_score=1.0,
                    variables_defined={var_name}
                ))
        
        return var_chunks
    
    def _calculate_enhanced_complexity(self, node: ast.AST) -> float:
        """Enhanced cyclomatic complexity calculation with additional metrics"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Traditional cyclomatic complexity factors
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            
            # Additional complexity factors
            elif isinstance(child, ast.ListComp):  # List comprehension
                complexity += 1
            elif isinstance(child, ast.DictComp):  # Dict comprehension
                complexity += 1
            elif isinstance(child, ast.Lambda):    # Lambda functions
                complexity += 1
        
        return float(complexity)
    
    def _extract_function_calls(self, node: ast.AST) -> Set[str]:
        """Extract all function calls within a node"""
        calls = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(f"{self._get_name(child.func.value)}.{child.func.attr}")
        
        return calls
    
    def _extract_variables(self, node: ast.AST) -> Set[str]:
        """Extract all variables defined within a node"""
        variables = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        variables.add(target.id)
        
        return variables
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name from AST node"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_name(decorator.value)}.{decorator.attr}"
        return str(decorator)
    
    def _get_name(self, node: ast.AST) -> str:
        """Safely extract name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"

class EnhancedEmbeddingGenerator:
    """Advanced embedding generation with multiple strategies"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Embedding cache for performance
        self.embedding_cache = {}
    
    def generate_embeddings(self, chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Generate embeddings with caching and batch processing"""
        if not chunks:
            return chunks
        
        # Prepare texts for embedding with enhanced context
        texts_to_embed = []
        cache_keys = []
        chunks_to_embed = []
        
        for chunk in chunks:
            enhanced_text = self._prepare_enhanced_text(chunk)
            cache_key = hashlib.md5(enhanced_text.encode()).hexdigest()
            
            if cache_key in self.embedding_cache:
                chunk.embedding = self.embedding_cache[cache_key]
            else:
                texts_to_embed.append(enhanced_text)
                cache_keys.append(cache_key)
                chunks_to_embed.append(chunk)
        
        # Generate embeddings for uncached texts
        if texts_to_embed:
            with st.spinner(f"🧠 Generating semantic embeddings for {len(texts_to_embed)} code chunks..."):
                progress_bar = st.progress(0)
                
                # Batch processing for better performance
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(texts_to_embed), batch_size):
                    batch_texts = texts_to_embed[i:i+batch_size]
                    batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                    all_embeddings.extend(batch_embeddings)
                    
                    progress = min((i + batch_size) / len(texts_to_embed), 1.0)
                    progress_bar.progress(progress)
                
                progress_bar.empty()
                
                # Assign embeddings and cache them
                for chunk, embedding, cache_key in zip(chunks_to_embed, all_embeddings, cache_keys):
                    chunk.embedding = embedding
                    self.embedding_cache[cache_key] = embedding
        
        return chunks
    
    def _prepare_enhanced_text(self, chunk: CodeChunk) -> str:
        """Enhanced text preparation with comprehensive context"""
        text_parts = [
            f"Type: {chunk.chunk_type}",
            f"Name: {chunk.name}",
        ]
        
        # Add complexity context
        text_parts.append(f"Complexity: {chunk.complexity_category}")
        
        # Add parameter information for functions
        if chunk.parameters:
            text_parts.append(f"Parameters: {', '.join(chunk.parameters)}")
        
        # Add decorator information
        if chunk.decorators:
            text_parts.append(f"Decorators: {', '.join(chunk.decorators)}")
        
        # Add docstring if available
        if chunk.docstring:
            # Limit docstring to first sentence for embedding
            first_sentence = chunk.docstring.split('.')[0][:200]
            text_parts.append(f"Description: {first_sentence}")
        
        # Add the actual code content
        text_parts.append(f"Content: {chunk.content}")
        
        return " ".join(text_parts)

class AdvancedVectorDatabase:
    """Enhanced vector database with multiple index types and advanced search"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        
        # Multiple index types for different use cases
        self.flat_index = faiss.IndexFlatIP(embedding_dim)  # Exact search
        self.quantized_index = None  # For large datasets
        self.hnsw_index = None      # For approximate search
        
        self.chunks = []
        self.is_built = False
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'popular_queries': Counter()
        }
    
    def add_chunks(self, chunks: List[CodeChunk]):
        """Add chunks to multiple index types"""
        if not chunks or not chunks[0].embedding is not None:
            return
        
        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to flat index
        self.flat_index.add(embeddings.astype('float32'))
        
        # For larger datasets, also create quantized index
        if len(chunks) > 1000:
            self.quantized_index = faiss.IndexPQ(self.embedding_dim, 8, 8)
            self.quantized_index.train(embeddings.astype('float32'))
            self.quantized_index.add(embeddings.astype('float32'))
        
        self.chunks.extend(chunks)
        self.is_built = True
    
    def search(self, query_embedding: np.ndarray, k: int = 5, 
              search_type: str = "exact") -> List[Tuple[CodeChunk, float]]:
        """Advanced search with multiple algorithms"""
        if not self.is_built:
            return []
        
        start_time = time.time()
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Choose search index based on type and data size
        if search_type == "exact" or self.quantized_index is None:
            distances, indices = self.flat_index.search(query_embedding, min(k, len(self.chunks)))
        else:
            distances, indices = self.quantized_index.search(query_embedding, min(k, len(self.chunks)))
        
        # Prepare results with additional metadata
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # Valid index
                similarity_score = float(distances[0][i])
                chunk = self.chunks[idx]
                results.append((chunk, similarity_score))
        
        # Update search statistics
        search_time = time.time() - start_time
        self.search_stats['total_searches'] += 1
        self.search_stats['avg_search_time'] = (
            (self.search_stats['avg_search_time'] * (self.search_stats['total_searches'] - 1) + search_time) /
            self.search_stats['total_searches']
        )
        
        return results
    
    def get_chunk_similarities(self) -> np.ndarray:
        """Compute similarity matrix between all chunks"""
        if not self.is_built:
            return np.array([])
        
        embeddings = np.array([chunk.embedding for chunk in self.chunks])
        return cosine_similarity(embeddings)

class ComprehensiveRAGSystem:
    """Advanced RAG system with comprehensive analysis capabilities"""
    
    def __init__(self):
        self.parser = AdvancedCodeParser()
        self.embedding_generator = EnhancedEmbeddingGenerator()
        self.vector_db = AdvancedVectorDatabase(self.embedding_generator.embedding_dim)
        self.chunks = []
        
        # Analysis results cache
        self.analysis_cache = {}
        self.last_analysis_hash = None
    
    def process_code(self, code: str, file_path: str = "input.py") -> Dict:
        """Process code through enhanced RAG pipeline"""
        # Generate hash for caching
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        if code_hash == self.last_analysis_hash:
            return self.analysis_cache.get(code_hash, {})
        
        # Step 1: Advanced parsing with comprehensive analysis
        with st.spinner("🔍 Parsing code structure..."):
            self.chunks = self.parser.parse_python_code(code, file_path)
        
        if not self.chunks:
            return {"error": "No analyzable code chunks found"}
        
        # Step 2: Generate semantic embeddings
        self.chunks = self.embedding_generator.generate_embeddings(self.chunks)
        
        # Step 3: Build vector database
        with st.spinner("🏗️ Building semantic search index..."):
            self.vector_db.add_chunks(self.chunks)
        
        # Step 4: Generate comprehensive analysis
        analysis_results = self._generate_comprehensive_analysis()
        
        # Cache results
        self.analysis_cache[code_hash] = analysis_results
        self.last_analysis_hash = code_hash
        
        return analysis_results
    
    def search_code(self, query: str, k: int = 5, filters: Dict = None) -> List[Dict]:
        """Enhanced semantic search with filtering and ranking"""
        if not self.vector_db.is_built:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.model.encode([query])[0]
        
        # Perform search
        results = self.vector_db.search(query_embedding, k * 2)  # Get more for filtering
        
        # Apply filters if provided
        if filters:
            results = self._apply_search_filters(results, filters)
        
        # Format and rank results
        formatted_results = []
        for chunk, similarity in results[:k]:
            formatted_results.append({
                "chunk": chunk,
                "similarity": similarity,
                "type": chunk.chunk_type,
                "name": chunk.name,
                "complexity": chunk.complexity_score,
                "quality_score": chunk.quality_score,
                "lines": f"{chunk.start_line}-{chunk.end_line}",
                "maintainability": chunk.maintainability_index,
                "parameters": chunk.parameters,
                "decorators": chunk.decorators
            })
        
        # Update search statistics
        self.vector_db.search_stats['popular_queries'][query] += 1
        
        return formatted_results
    
    def _apply_search_filters(self, results: List[Tuple[CodeChunk, float]], 
                            filters: Dict) -> List[Tuple[CodeChunk, float]]:
        """Apply various filters to search results"""
        filtered_results = []
        
        for chunk, similarity in results:
            # Type filter
            if 'types' in filters and chunk.chunk_type not in filters['types']:
                continue
            
            # Complexity filter
            if 'max_complexity' in filters and chunk.complexity_score > filters['max_complexity']:
                continue
            
            # Quality filter
            if 'min_quality' in filters and chunk.quality_score < filters['min_quality']:
                continue
            
            filtered_results.append((chunk, similarity))
        
        return filtered_results
    
    def _generate_comprehensive_analysis(self) -> Dict:
        """Generate comprehensive codebase analysis"""
        if not self.chunks:
            return {}
        
        analysis = {
            "basic_stats": self._calculate_basic_statistics(),
            "complexity_analysis": self._analyze_complexity_patterns(),
            "quality_metrics": self._analyze_code_quality(),
            "structure_analysis": self._analyze_code_structure(),
            "semantic_clustering": self._perform_semantic_clustering(),
            "dependency_analysis": self._analyze_dependencies(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _calculate_basic_statistics(self) -> Dict:
        """Calculate comprehensive basic statistics"""
        stats = {
            "total_chunks": len(self.chunks),
            "chunk_types": Counter(c.chunk_type for c in self.chunks),
            "total_lines": sum(c.lines_of_code for c in self.chunks),
            "avg_lines_per_chunk": np.mean([c.lines_of_code for c in self.chunks]),
            "functions_count": sum(1 for c in self.chunks if c.chunk_type in ['function', 'method', 'async_function']),
            "classes_count": sum(1 for c in self.chunks if c.chunk_type == 'class'),
            "imports_count": sum(1 for c in self.chunks if c.chunk_type == 'import')
        }
        
        return stats
    
    def _analyze_complexity_patterns(self) -> Dict:
        """Analyze complexity patterns and distributions"""
        complexities = [c.complexity_score for c in self.chunks]
        
        return {
            "mean": np.mean(complexities),
            "std": np.std(complexities),
            "median": np.median(complexities),
            "max": np.max(complexities),
            "min": np.min(complexities),
            "distribution": {
                "low": sum(1 for c in complexities if c <= 5),
                "medium": sum(1 for c in complexities if 5 < c <= 10),
                "high": sum(1 for c in complexities if 10 < c <= 20),
                "very_high": sum(1 for c in complexities if c > 20)
            }
        }
    
    def _analyze_code_quality(self) -> Dict:
        """Analyze overall code quality metrics"""
        quality_scores = [c.quality_score for c in self.chunks]
        maintainability_scores = [c.maintainability_index for c in self.chunks]
        comment_ratios = [c.comment_ratio for c in self.chunks]
        
        return {
            "avg_quality_score": np.mean(quality_scores),
            "avg_maintainability": np.mean(maintainability_scores),
            "avg_comment_ratio": np.mean(comment_ratios),
            "high_quality_chunks": sum(1 for q in quality_scores if q > 0.8),
            "needs_improvement": sum(1 for q in quality_scores if q < 0.5)
        }
    
    def _analyze_code_structure(self) -> Dict:
        """Analyze code structure and organization"""
        return {
            "has_classes": any(c.chunk_type == 'class' for c in self.chunks),
            "has_async_functions": any(c.chunk_type == 'async_function' for c in self.chunks),
            "decorators_used": set().union(*[c.decorators for c in self.chunks if c.decorators]),
            "imported_modules": set().union(*[c.imports_used for c in self.chunks if hasattr(c, "imports_used") and c.imports_used]),
            "methods_per_class": self._methods_per_class(),
            "functions_per_file": self._functions_per_file(),
            "variables_defined": set().union(*[c.variables_defined for c in self.chunks if hasattr(c, "variables_defined") and c.variables_defined])
        }

    def _methods_per_class(self) -> Dict[str, int]:
        """Count methods per class"""
        class_methods = {}
        for c in self.chunks:
            if c.chunk_type == "method" and "." in c.name:
                class_name = c.name.split(".", 1)[0]
                class_methods[class_name] = class_methods.get(class_name, 0) + 1
        return class_methods

    def _functions_per_file(self) -> int:
        """Count functions per file (assuming single file for now)"""
        return sum(1 for c in self.chunks if c.chunk_type in ["function", "async_function", "method"])

    def _perform_semantic_clustering(self) -> Dict:
        """Cluster code chunks using KMeans and reduce dimensions for visualization"""
        if not self.chunks or not any(c.embedding is not None for c in self.chunks):
            return {}

        embeddings = np.array([c.embedding for c in self.chunks if c.embedding is not None])
        n_clusters = min(8, len(embeddings))
        if n_clusters < 2:
            return {}

        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)

        # Prepare cluster info
        clusters = []
        for idx, c in enumerate(self.chunks):
            if c.embedding is not None:
                clusters.append({
                    "name": c.name,
                    "type": c.chunk_type,
                    "cluster": int(cluster_labels[idx]),
                    "x": float(reduced[idx, 0]),
                    "y": float(reduced[idx, 1])
                })

        return {
            "clusters": clusters,
            "n_clusters": n_clusters,
            "explained_variance": float(np.sum(pca.explained_variance_ratio_))
        }

    def _analyze_dependencies(self) -> Dict:
        """Analyze import and call dependencies"""
        import_graph = dict(self.parser.import_graph)
        call_graph = dict(self.parser.call_graph)
        inheritance_graph = dict(self.parser.inheritance_graph)
        return {
            "import_graph": import_graph,
            "call_graph": call_graph,
            "inheritance_graph": inheritance_graph
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate code improvement recommendations based on analysis"""
        recs = []
        quality = self._analyze_code_quality()
        complexity = self._analyze_complexity_patterns()
        stats = self._calculate_basic_statistics()

        if quality["avg_quality_score"] < 0.6:
            recs.append("Increase code comments and improve maintainability for better quality.")
        if complexity["max"] > 20:
            recs.append("Refactor highly complex functions/classes to reduce cyclomatic complexity.")
        if stats["functions_count"] > 0 and stats["classes_count"] == 0:
            recs.append("Consider using classes for better code organization and reusability.")
        if stats["imports_count"] > 10:
            recs.append("Review import statements for possible redundancy or unused imports.")
        if quality["needs_improvement"] > 0:
            recs.append(f"{quality['needs_improvement']} code chunks have low quality scores. Review and refactor as needed.")

        if not recs:
            recs.append("Codebase is well-structured and maintains good quality standards.")

        return recs

# --- Streamlit UI and main logic would follow here ---

def main():
    st.markdown('<div class="main-header"><h1>🧠 Advanced RAG Code Intelligence System</h1></div>', unsafe_allow_html=True)
    
    EXAMPLE_CODE = '''
def add(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    def multiply(self, x, y):
        # Multiplies two numbers
        return x * y
'''

    st.sidebar.header("Upload & Settings")

    uploaded_file = st.sidebar.file_uploader("Upload Python file", type=["py"])
    example_code = st.sidebar.checkbox("Use example code", value=False)
    search_query = st.sidebar.text_input("Semantic Search Query", "")
    k_results = st.sidebar.slider("Top-K Results", 1, 10, 5)
    show_analytics = st.sidebar.checkbox("Show Analytics", value=True)
    show_call_graph = st.sidebar.checkbox("Show Call Graph", value=False)  # Advanced feature

    # --- Add main tabs for new features ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "📝 Code Input & Search", 
        "📊 Code Analytics", 
        "🎯 System Demo", 
        "⚙️ How It Works",
        "📈 Trend Analysis", # New Tab
        "🌐 Dependency Graph", # New Tab
        "🔮 Predictive Insights", # New Tab
        "🔄 Refactoring Suggestions", # New Tab
        "🛡️ Security Analysis", # New Tab
        "📚 Documentation Generator", # New Tab
        "🧪 Test Coverage", # New Tab
        "📊 Custom Reports", # New Tab
    ])

    with tab1:
        # --- Original code input, analytics, and search UI ---
        if uploaded_file or example_code:
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"

            rag = ComprehensiveRAGSystem()
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Code Overview")
            st.code(code, language="python")

            if show_analytics:
                st.subheader("📊 Codebase Analytics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chunks", analysis["basic_stats"]["total_chunks"])
                    st.metric("Functions", analysis["basic_stats"]["functions_count"])
                with col2:
                    st.metric("Classes", analysis["basic_stats"]["classes_count"])
                    st.metric("Imports", analysis["basic_stats"]["imports_count"])
                with col3:
                    st.metric("Avg. Lines/Chunk", f"{analysis['basic_stats']['avg_lines_per_chunk']:.1f}")
                    st.metric("Total Lines", analysis["basic_stats"]["total_lines"])

                st.markdown("#### Complexity Distribution")
                complexity_dist = analysis["complexity_analysis"]["distribution"]
                st.bar_chart(pd.DataFrame.from_dict(complexity_dist, orient="index", columns=["Count"]))

                # --- Advanced Feature: Complexity Heatmap ---
                st.markdown("#### Code Complexity Heatmap")
                chunk_names = [c.name for c in rag.chunks]
                complexities = [c.complexity_score for c in rag.chunks]
                if chunk_names and complexities:
                    heatmap_df = pd.DataFrame({
                        "Chunk": chunk_names,
                        "Complexity": complexities
                    })
                    fig = px.density_heatmap(
                        heatmap_df, x="Chunk", y="Complexity", nbinsy=10, color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Quality Metrics")
                st.json(analysis["quality_metrics"])

                st.markdown("#### Structure Analysis")
                st.json(analysis["structure_analysis"])

                st.markdown("#### Recommendations")
                for rec in analysis["recommendations"]:
                    st.info(rec)

                # --- Advanced Feature: Download Analysis Report ---
                st.markdown("#### Download Analysis Report")
                report_json = json.dumps(analysis, indent=2, default=str)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name="code_analysis_report.json",
                    mime="application/json"
                )

                if analysis["semantic_clustering"]:
                    st.markdown("#### Semantic Clustering (2D PCA)")
                    clusters = pd.DataFrame(analysis["semantic_clustering"]["clusters"])
                    fig = px.scatter(
                        clusters, x="x", y="y", color="cluster", hover_data=["name", "type"],
                        title="Semantic Clusters of Code Chunks"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # --- Advanced Feature: Call Graph Visualization ---
                if show_call_graph:
                    st.markdown("#### Call Graph Visualization")
                    call_graph = analysis["dependency_analysis"]["call_graph"]
                    if call_graph:
                        G = nx.DiGraph()
                        for caller, callees in call_graph.items():
                            for callee in callees:
                                G.add_edge(caller, callee)
                        pos = nx.spring_layout(G, k=0.5, iterations=20, seed=42)
                        edge_x = []
                        edge_y = []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x += [x0, x1, None]
                            edge_y += [y0, y1, None]
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1, color='#888'),
                            hoverinfo='none',
                            mode='lines'
                        )
                        node_x = []
                        node_y = []
                        node_text = []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(node)
                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition="top center",
                            marker=dict(size=20, color='rgba(102,126,234,0.8)'),
                            hoverinfo='text'
                        )
                        fig = go.Figure(data=[edge_trace, node_trace],
                                        layout=go.Layout(
                                            showlegend=False,
                                            hovermode='closest',
                                            margin=dict(b=20,l=5,r=5,t=40),
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            title="Function/Method Call Graph"
                                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No call relationships detected.")

            if search_query:
                st.subheader(f"🔎 Semantic Search Results for: '{search_query}'")
                results = rag.search_code(search_query, k=k_results)
                for res in results:
                    chunk = res["chunk"]
                    # --- Advanced Feature: Expandable Chunk Details ---
                    with st.expander(f"{chunk.name} ({chunk.chunk_type}, Complexity: {chunk.complexity_category})"):
                        st.markdown(
                            f"<div class='metric-card'><b>{chunk.name}</b> "
                            f"({chunk.chunk_type}, Complexity: <span class='complexity-indicator complexity-{chunk.complexity_category.lower()}'>{chunk.complexity_category}</span>)<br>"
                            f"<pre>{chunk.content}</pre>"
                            f"<small>Similarity: {res['similarity']:.3f} | Quality: {res['quality_score']:.2f} | Lines: {res['lines']}</small></div>",
                            unsafe_allow_html=True
                        )
                        st.markdown("**Parameters:** " + ", ".join(chunk.parameters) if chunk.parameters else "_None_")
                        st.markdown("**Decorators:** " + ", ".join(chunk.decorators) if chunk.decorators else "_None_")
                        st.markdown("**Docstring:**")
                        st.code(chunk.docstring or "_None_", language="markdown")
                        st.markdown(f"**Maintainability Index:** {chunk.maintainability_index:.2f}")
                        st.markdown(f"**Comment Ratio:** {chunk.comment_ratio:.2f}")
                        st.markdown(f"**Variables Defined:** {', '.join(chunk.variables_defined) if chunk.variables_defined else '_None_'}")
                        st.markdown(f"**Calls Made:** {', '.join(chunk.calls_made) if chunk.calls_made else '_None_'}")
        else:
            st.info("Upload a Python file or use example code to get started.")

    with tab2:
        st.header("📊 Code Analytics & Insights")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)
            
            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Code Quality Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maintainability Index", 
                        f"{analysis.get('quality_metrics', {}).get('maintainability_index', 0):.2f}")
                st.metric("Comment Ratio", 
                        f"{analysis.get('quality_metrics', {}).get('comment_ratio', 0):.2%}")
            with col2:
                st.metric("Complexity Score", 
                        f"{analysis.get('quality_metrics', {}).get('complexity_score', 0):.2f}")
                st.metric("Code Coverage", 
                        f"{analysis.get('quality_metrics', {}).get('code_coverage', 0):.2%}")
            with col3:
                st.metric("Bug Risk", 
                        f"{analysis.get('quality_metrics', {}).get('bug_risk', 0):.2%}")
                st.metric("Technical Debt", 
                        f"{analysis.get('quality_metrics', {}).get('technical_debt', 0):.2%}")

            st.subheader("Code Structure Analysis")
            if "structure_analysis" in analysis:
                st.json(analysis["structure_analysis"], expanded=False)
            else:
                st.warning("Structure analysis not available")

            st.subheader("Dependency Analysis")
            if "dependency_analysis" in analysis and "graphviz" in analysis["dependency_analysis"]:
                st.graphviz_chart(analysis["dependency_analysis"]["graphviz"])
            else:
                st.warning("Dependency analysis graph not available")

            st.subheader("Performance Metrics")
            if "performance_metrics" in analysis and analysis["performance_metrics"]:
                try:
                    perf_df = pd.DataFrame(analysis["performance_metrics"])
                    st.bar_chart(perf_df.set_index("metric"))
                except Exception as e:
                    st.warning(f"Could not display performance metrics: {str(e)}")
            else:
                st.warning("Performance metrics not available")

            st.subheader("Code Evolution Trends")
            if analysis.get("trend_analysis"):
                try:
                    trend_df = pd.DataFrame(analysis["trend_analysis"])
                    if not trend_df.empty:
                        st.line_chart(trend_df)
                    else:
                        st.info("No trend data available")
                except Exception as e:
                    st.warning(f"Could not display trend data: {str(e)}")
            else:
                st.info("No historical data available for trend analysis")
        else:
            st.info("Upload a Python file or use example code to see analytics.")

    with tab5:
        st.header("📈 Code Trend Analysis")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Metric Trends Over Time")
            if analysis["trend_analysis"]:
                trend_df = pd.DataFrame(analysis["trend_analysis"])
                
                # Create multiple charts
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(px.line(trend_df, x="date", y="complexity_score", 
                                          title="Complexity Score Trend"))
                    st.plotly_chart(px.line(trend_df, x="date", y="maintainability_index", 
                                          title="Maintainability Index Trend"))
                with col2:
                    st.plotly_chart(px.line(trend_df, x="date", y="code_coverage", 
                                          title="Code Coverage Trend"))
                    st.plotly_chart(px.line(trend_df, x="date", y="bug_risk", 
                                          title="Bug Risk Trend"))
            else:
                st.info("No historical data available for trend analysis")

            st.subheader("Code Evolution Heatmap")
            if analysis["evolution_heatmap"]:
                heatmap_df = pd.DataFrame(analysis["evolution_heatmap"])
                fig = px.density_heatmap(
                    heatmap_df, x="date", y="metric", z="value",
                    title="Code Evolution Heatmap",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig)
        else:
            st.info("Upload a Python file or use example code to see trend analysis.")

    with tab6:
        st.header("🌐 Interactive Dependency Graph")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Call Graph Visualization")
            if analysis.get("dependency_analysis", {}).get("call_graph"):
                call_graph = analysis["dependency_analysis"]["call_graph"]
                G = nx.DiGraph()
                
                try:
                    # Add nodes and edges
                    for caller, callees in call_graph.items():
                        if caller:  # Ensure caller is not None or empty
                            G.add_node(str(caller), type="function")
                            for callee in callees:
                                if callee:  # Ensure callee is not None or empty
                                    G.add_node(str(callee), type="function")
                                    G.add_edge(str(caller), str(callee))
                    
                    # Create interactive visualization
                    pos = nx.spring_layout(G)
                    edge_x = []
                    edge_y = []
                    for edge in G.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x += [x0, x1, None]
                        edge_y += [y0, y1, None]
                    
                    edge_trace = go.Scatter(
                        x=edge_x, y=edge_y,
                        line=dict(width=1, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    node_x = []
                    node_y = []
                    node_text = []
                    node_size = []
                    for node in G.nodes():
                        x, y = pos[node]
                        node_x.append(x)
                        node_y.append(y)
                        node_text.append(node)
                        # Node size based on number of connections
                        node_size.append(len(list(G.neighbors(node))) * 10)
                    
                    node_trace = go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        text=node_text,
                        textposition="top center",
                        marker=dict(
                            size=node_size,
                            color='rgba(102,126,234,0.8)',
                            sizemode='area'
                        ),
                        hoverinfo='text'
                    )
                    
                    fig = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20,l=5,r=5,t=40),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      title="Function Call Graph"
                                  ))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render call graph: {str(e)}")
            else:
                st.info("No call relationships detected in the code.")

            st.subheader("Import Dependency Tree")
            if analysis.get("dependency_analysis", {}).get("import_tree"):
                import_tree = analysis["dependency_analysis"]["import_tree"]
                try:
                    st.json(import_tree, expanded=False)
                except Exception as e:
                    st.warning(f"Could not display import tree: {str(e)}")
            else:
                st.info("No import dependency data available")
        else:
            st.info("Upload a Python file or use example code to see dependency graphs.")

    with tab7:
        st.header("🔮 Predictive Code Insights")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Predictive Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Future Bug Risk", f"{analysis['predictive_metrics']['future_bug_risk']:.2%}")
                st.metric("Technical Debt Growth", f"{analysis['predictive_metrics']['technical_debt_growth']:.2%}")
            with col2:
                st.metric("Maintenance Effort", f"{analysis['predictive_metrics']['maintenance_effort']:.2f} hours")
                st.metric("Refactoring Priority", analysis['predictive_metrics']['refactoring_priority'])

            st.subheader("Risk Hotspots")
            if analysis["risk_hotspots"]:
                risk_df = pd.DataFrame(analysis["risk_hotspots"])
                st.dataframe(risk_df.style.background_gradient(cmap='Reds'))

            st.subheader("Development Effort Estimation")
            if analysis["effort_estimation"]:
                effort_df = pd.DataFrame(analysis["effort_estimation"])
                st.bar_chart(effort_df.set_index("component"))

            st.subheader("Refactoring Opportunities")
            if analysis["refactoring_opportunities"]:
                for opportunity in analysis["refactoring_opportunities"]:
                    with st.expander(opportunity["name"]):
                        st.markdown(f"**Type:** {opportunity['type']}")
                        st.markdown(f"**Priority:** {opportunity['priority']}")
                        st.markdown(f"**Description:** {opportunity['description']}")
                        st.markdown(f"**Estimated Savings:** {opportunity['estimated_savings']}")
        else:
            st.info("Upload a Python file or use example code to see predictive insights.")

    with tab8:
        st.header("🔄 Refactoring Suggestions")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Refactoring Opportunities")
            if analysis["refactoring_suggestions"]:
                for suggestion in analysis["refactoring_suggestions"]:
                    with st.expander(suggestion["title"]):
                        st.markdown(f"**Type:** {suggestion['type']}")
                        st.markdown(f"**Priority:** {suggestion['priority']}")
                        st.markdown(f"**Current Code:**")
                        st.code(suggestion["current_code"], language="python")
                        st.markdown(f"**Suggested Changes:**")
                        st.code(suggestion["suggested_code"], language="python")
                        st.markdown(f"**Benefits:**")
                        st.markdown(suggestion["benefits"])
                        st.markdown(f"**Impact:**")
                        st.markdown(suggestion["impact"])

            st.subheader("Code Duplication Analysis")
            if analysis["code_duplicates"]:
                dup_df = pd.DataFrame(analysis["code_duplicates"])
                st.dataframe(dup_df.style.background_gradient(cmap='Blues'))

            st.subheader("Complexity Hotspots")
            if analysis["complexity_hotspots"]:
                hotspot_df = pd.DataFrame(analysis["complexity_hotspots"])
                st.dataframe(hotspot_df.style.background_gradient(cmap='Reds'))

            st.subheader("Code Style Improvements")
            if analysis["style_improvements"]:
                for improvement in analysis["style_improvements"]:
                    with st.expander(improvement["title"]):
                        st.markdown(f"**Current Implementation:**")
                        st.code(improvement["current_code"], language="python")
                        st.markdown(f"**Suggested Style:**")
                        st.code(improvement["suggested_code"], language="python")
                        st.markdown(f"**Rationale:**")
                        st.markdown(improvement["rationale"])
        else:
            st.info("Upload a Python file or use example code to see refactoring suggestions.")

    with tab9:
        st.header("🛡️ Security Analysis")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Security Risk Assessment")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Security Score", f"{analysis['security_metrics']['score']:.2f}")
                st.metric("Critical Vulnerabilities", analysis['security_metrics']['critical_count'])
            with col2:
                st.metric("High Risk Areas", analysis['security_metrics']['high_risk_count'])
                st.metric("Security Debt", f"{analysis['security_metrics']['security_debt']:.2%}")

            st.subheader("Security Vulnerabilities")
            if analysis["security_vulnerabilities"]:
                vuln_df = pd.DataFrame(analysis["security_vulnerabilities"])
                st.dataframe(vuln_df.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x == 'Critical' else ''
                ))

            st.subheader("Security Code Review")
            if analysis["security_review"]:
                for review in analysis["security_review"]:
                    with st.expander(review["title"]):
                        st.markdown(f"**Severity:** {review['severity']}")
                        st.markdown(f"**Type:** {review['type']}")
                        st.markdown(f"**Location:** {review['location']}")
                        st.markdown(f"**Description:**")
                        st.code(review["code_snippet"], language="python")
                        st.markdown(f"**Recommendation:**")
                        st.code(review["recommendation"], language="python")

            st.subheader("Dependency Security")
            if analysis["dependency_security"]:
                dep_df = pd.DataFrame(analysis["dependency_security"])
                st.dataframe(dep_df.style.applymap(
                    lambda x: 'background-color: #ffcccc' if x == 'Vulnerable' else ''
                ))
        else:
            st.info("Upload a Python file or use example code to see security analysis.")

    with tab10:
        st.header("📚 Documentation Generator")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Documentation Status")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Documentation", f"{analysis['documentation_metrics']['overall_coverage']:.2%}")
                st.metric("Missing Docstrings", analysis['documentation_metrics']['missing_count'])
            with col2:
                st.metric("Function Coverage", f"{analysis['documentation_metrics']['function_coverage']:.2%}")
                st.metric("Class Coverage", f"{analysis['documentation_metrics']['class_coverage']:.2%}")

            st.subheader("Generated Documentation")
            if analysis["generated_docs"]:
                for doc in analysis["generated_docs"]:
                    with st.expander(doc["title"]):
                        st.markdown(f"**Description:**")
                        st.markdown(doc["description"])
                        st.markdown(f"**Parameters:**")
                        st.json(doc["parameters"], expanded=False)
                        st.markdown(f"**Returns:**")
                        st.markdown(doc["returns"])
                        st.markdown(f"**Examples:**")
                        st.code(doc["examples"], language="python")

            st.subheader("Missing Documentation")
            if analysis["missing_docs"]:
                missing_df = pd.DataFrame(analysis["missing_docs"])
                st.dataframe(missing_df.style.background_gradient(cmap='Reds'))

            st.subheader("Documentation Quality")
            if analysis["doc_quality"]:
                quality_df = pd.DataFrame(analysis["doc_quality"])
                st.dataframe(quality_df.style.background_gradient(cmap='Greens'))

            st.subheader("Export Documentation")
            if analysis["export_docs"]:
                for format in analysis["export_docs"]:
                    st.download_button(
                        label=f"Download {format} Documentation",
                        data=analysis["export_docs"][format],
                        file_name=f"documentation.{format.lower()}"
                    )
        else:
            st.info("Upload a Python file or use example code to see documentation generation.")

    with tab11:
        st.header("🧪 Test Coverage Analysis")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Test Coverage Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Overall Coverage", f"{analysis['test_metrics']['overall_coverage']:.2%}")
                st.metric("Function Coverage", f"{analysis['test_metrics']['function_coverage']:.2%}")
            with col2:
                st.metric("Branch Coverage", f"{analysis['test_metrics']['branch_coverage']:.2%}")
                st.metric("Statement Coverage", f"{analysis['test_metrics']['statement_coverage']:.2%}")

            st.subheader("Coverage by Component")
            if analysis["coverage_by_component"]:
                coverage_df = pd.DataFrame(analysis["coverage_by_component"])
                st.bar_chart(coverage_df.set_index("component"))

            st.subheader("Uncovered Code")
            if analysis["uncovered_code"]:
                uncovered_df = pd.DataFrame(analysis["uncovered_code"])
                st.dataframe(uncovered_df.style.background_gradient(cmap='Reds'))

            st.subheader("Test Quality Analysis")
            if analysis["test_quality"]:
                quality_df = pd.DataFrame(analysis["test_quality"])
                st.dataframe(quality_df.style.background_gradient(cmap='Greens'))

            st.subheader("Coverage Trends")
            if analysis["coverage_trends"]:
                trend_df = pd.DataFrame(analysis["coverage_trends"])
                st.line_chart(trend_df)
        else:
            st.info("Upload a Python file or use example code to see test coverage analysis.")

    with tab12:
        st.header("📊 Custom Reports")
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)

            if "error" in analysis:
                st.error(analysis["error"])
                return

            st.subheader("Report Configuration")
            # Add filters and selections
            selected_metrics = st.multiselect(
                "Select Metrics to Include",
                options=["complexity", "quality", "security", "documentation", "test_coverage"],
                default=["complexity", "quality"]
            )

            selected_components = st.multiselect(
                "Select Components to Analyze",
                options=["functions", "classes", "modules", "dependencies"],
                default=["functions", "classes"]
            )

            report_format = st.selectbox(
                "Select Report Format",
                options=["PDF", "CSV", "JSON", "HTML"]
            )

            if st.button("Generate Report"):
                # Generate report based on selections
                report = analysis["custom_report"](
                    metrics=selected_metrics,
                    components=selected_components
                )
                
                # Display report preview
                st.subheader("Report Preview")
                if report_format == "JSON":
                    st.json(report)
                elif report_format == "CSV":
                    st.dataframe(pd.DataFrame(report))
                else:
                    st.markdown(report)

                # Download button
                report_data = json.dumps(report, indent=2)
                st.download_button(
                    label=f"Download {report_format} Report",
                    data=report_data,
                    file_name=f"code_analysis_report.{report_format.lower()}"
                )

            st.subheader("Saved Reports")
            if analysis["saved_reports"]:
                for report in analysis["saved_reports"]:
                    with st.expander(report["name"]):
                        st.markdown(f"**Date:** {report['date']}")
                        st.markdown(f"**Metrics:** {', '.join(report['metrics'])}")
                        st.markdown(f"**Components:** {', '.join(report['components'])}")
                        st.download_button(
                            label="Download Report",
                            data=report["data"],
                            file_name=report["name"]
                        )
        else:
            st.info("Upload a Python file or use example code to see custom reports.")
        # --- Show analytics if code is processed ---
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)
            if "error" in analysis:
                st.error(analysis["error"])
            else:
                st.subheader("Complexity Analysis")
                stats = analysis["complexity_analysis"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean", f"{stats['mean']:.2f}")
                with col2:
                    st.metric("Median", f"{stats['median']:.2f}")
                with col3:
                    st.metric("Max", f"{stats['max']:.0f}")
                with col4:
                    st.metric("Min", f"{stats['min']:.0f}")

                st.subheader("Quality Metrics")
                st.json(analysis["quality_metrics"])

                st.subheader("Structure Patterns")
                st.json(analysis["structure_analysis"])

                # --- Advanced Feature: Complexity Heatmap in Analytics Tab ---
                st.markdown("#### Code Complexity Heatmap")
                chunk_names = [c.name for c in rag.chunks]
                complexities = [c.complexity_score for c in rag.chunks]
                if chunk_names and complexities:
                    heatmap_df = pd.DataFrame({
                        "Chunk": chunk_names,
                        "Complexity": complexities
                    })
                    fig = px.density_heatmap(
                        heatmap_df, x="Chunk", y="Complexity", nbinsy=10, color_continuous_scale="Viridis",title="Density heatmap analysis"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                if analysis["semantic_clustering"]:
                    st.subheader("Semantic Clustering Visualization")
                    clusters = pd.DataFrame(analysis["semantic_clustering"]["clusters"])
                    fig = px.scatter(
                        clusters, x="x", y="y", color="cluster", hover_data=["name", "type"],
                        title="Semantic Clusters of Code Chunks (analysis)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                # --- Advanced Feature: Call Graph Visualization in Analytics Tab ---
                if show_call_graph:
                    st.markdown("#### Call Graph Visualization")
                    call_graph = analysis["dependency_analysis"]["call_graph"]
                    if call_graph:
                        G = nx.DiGraph()
                        for caller, callees in call_graph.items():
                            for callee in callees:
                                G.add_edge(caller, callee)
                        pos = nx.spring_layout(G, k=0.5, iterations=20, seed=42)
                        edge_x = []
                        edge_y = []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x += [x0, x1, None]
                            edge_y += [y0, y1, None]
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1, color='#888'),
                            hoverinfo='none',
                            mode='lines'
                        )
                        node_x = []
                        node_y = []
                        node_text = []
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(node)
                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=node_text,
                            textposition="top center",
                            marker=dict(size=20, color='rgba(102,126,234,0.8)'),
                            hoverinfo='text'
                        )
                        fig = go.Figure(data=[edge_trace, node_trace],
                                        layout=go.Layout(
                                            showlegend=False,
                                            hovermode='closest',
                                            margin=dict(b=20,l=5,r=5,t=40),
                                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                            title="Function/Method Call Graph"
                                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No call relationships detected.")
        else:
            st.info("Process some code in the first tab to see analytics.")

    with tab3:
        st.header("🎯 RAG System Architecture Demo")
        st.markdown("""
        This application demonstrates a complete **Retrieval-Augmented Generation (RAG)** system 
        specifically designed for code analysis, incorporating the same advanced techniques used in 
        AI coding assistants like GitHub Copilot.
        """)
        st.subheader("System Architecture")
        # --- Simple flow diagram using Plotly ---
        import plotly.graph_objects as go
        steps = [
            "Code Input", "AST Parsing", "Chunk Extraction", 
            "Embedding Generation", "Vector Storage", "Semantic Search", "Result Generation"
        ]
        fig = go.Figure()
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
        # Ensure unique chart rendering
        st.plotly_chart(fig, use_container_width=True, key="rag_pipeline_flow")
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
        # --- Optionally, show stats if code processed ---
        if uploaded_file or example_code:
            rag = ComprehensiveRAGSystem()
            if uploaded_file:
                code = uploaded_file.read().decode("utf-8")
                file_path = uploaded_file.name
            else:
                code = EXAMPLE_CODE
                file_path = "example.py"
            analysis = rag.process_code(code, file_path)
            if "basic_stats" in analysis:
                st.subheader("Current Session Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Processed Chunks", analysis["basic_stats"]["total_chunks"])
                with col2:
                    st.metric("Functions", analysis["basic_stats"]["functions_count"])
                with col3:
                    st.metric("Classes", analysis["basic_stats"]["classes_count"])
            elif "error" in analysis:
                st.error(analysis["error"])

    with tab4:
        st.header("⚙️ How It Works: A Deep Dive into the RAG System")

        st.markdown("""
        This tab provides a **super comprehensive, step-by-step explanation** of the technologies and processes used in this application.
        Use this as a guide to understand how we turn raw code into a searchable, analyzable knowledge base.

        ---
        """)

        with st.expander("🔎 **Overview: What is a RAG Code Intelligence System?**", expanded=True):
            st.markdown("""
            The **Retrieval-Augmented Generation (RAG) Code Intelligence System** is a multi-stage pipeline that combines:
            - **Traditional program analysis** (AST parsing, code metrics)
            - **Modern machine learning** (transformer-based embeddings)
            - **Advanced information retrieval** (vector search, clustering, analytics)

            This system is inspired by tools like GitHub Copilot and is designed to help developers, reviewers, and engineering leaders **understand, search, and improve code at scale**.

            ---
            **Key Stages (Think of it like a Library!):**
            1. **Parsing & Chunking**: Like breaking a book into chapters and paragraphs.
            2. **Embedding**: Turning each paragraph into a unique "fingerprint" that captures its meaning.
            3. **Indexing & Storage**: Organizing all fingerprints in a super-fast filing cabinet.
            4. **Retrieval & Search**: Asking questions and instantly finding the most relevant paragraphs.
            5. **Analytics & Insights**: Summarizing, clustering, and visualizing the whole library.

            ---
            **Why RAG for Code?**
            - **Bridges the gap** between static code analysis and semantic understanding.
            - Enables **natural language search** and contextual recommendations.
            - Scales to large codebases and supports continuous integration workflows.

            **Real-World Use Cases:**
            - Code review automation and quality gates.
            - Onboarding new developers with codebase exploration.
            - Refactoring and technical debt identification.
            - AI-powered documentation and code summarization.

            ---
            **Did you know?**
            The same core ideas behind RAG are used in search engines, chatbots, and even self-driving cars (for retrieving relevant knowledge)!
            """)

        with st.expander("1️⃣ Parsing & Chunking - Understanding Code Structure", expanded=False):
            st.markdown("""
            **Objective:**
            Break down code into logical, meaningful units ("chunks") such as functions, classes, imports, and variables.

            **How it works:**
            - Uses Python's `ast` (Abstract Syntax Tree) module to parse code into a tree structure.
            - Walks the AST to extract:
                - Function definitions (with parameters, decorators, docstrings, etc.)
                - Class definitions (with inheritance, methods, properties)
                - Import statements
                - Module-level variables and constants
            - Each chunk is annotated with rich metadata: name, type, start/end lines, complexity, comments, etc.

            ---
            **Analogy:**
            Imagine reading a recipe book and making a list of all the ingredients, steps, and tips for each recipe. That's what parsing and chunking does for code!

            **Advanced Details:**
            - **Handles nested structures:** Methods inside classes, inner functions, and async functions.
            - **Tracks relationships:** Builds call graphs, import graphs, and inheritance hierarchies.
            - **Extracts quality metrics:** Cyclomatic complexity, maintainability index, comment ratio, and more.
            - **Supports extensibility:** Can be adapted for other languages or custom code patterns.

            **Why is this important?**
            - **Precision:** Extracts semantically complete units, not just lines or blocks.
            - **Rich Metadata:** Enables advanced analytics and search.
            - **Extensibility:** Can be extended to extract call graphs, inheritance, and more.

            **Example:**
            ```python
            import ast
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Extract function details
                    ...
                elif isinstance(node, ast.ClassDef):
                    # Extract class details
                    ...
            ```
            """)
            st.info("**Tip:** ASTs are like the X-ray vision for code—they reveal the structure beneath the surface!")

            with st.expander("🔬 Advanced: What is an AST and why use it?"):
                st.markdown("""
                - An **Abstract Syntax Tree (AST)** is a tree representation of the source code structure.
                - Each node represents a construct (function, class, assignment, etc.).
                - AST parsing is robust to formatting, whitespace, and comments.
                - Enables static analysis, code transformation, and deep inspection.

                **Technical Note:**
                ASTs are the foundation for many tools: linters, formatters, refactoring engines, and even compilers.
                """)

            with st.expander("🧮 Metrics Extracted at this Stage"):
                st.markdown("""
                - **Cyclomatic Complexity**: Measures the number of independent paths through code.
                - **Comment Ratio**: Fraction of lines that are comments.
                - **Maintainability Index**: Composite score based on complexity, comments, and length.
                - **Chunk Type Distribution**: Counts of functions, classes, imports, etc.
                - **Call Graphs**: Maps which functions/methods call each other.
                - **Inheritance Graphs**: Shows class inheritance relationships.
                """)

            with st.expander("💡 Real-World Example"):
                st.markdown("""
                - **Detecting dead code:** By analyzing call graphs, unused functions can be flagged.
                - **API surface extraction:** Quickly list all public classes and methods for documentation.
                """)

        with st.expander("2️⃣ Embedding - Turning Code into Semantic Vectors", expanded=False):
            st.markdown("""
            **Objective:**
            Convert code chunks into high-dimensional vectors ("embeddings") that capture their semantic meaning.

            **How it works:**
            - Uses a pre-trained transformer model (`all-MiniLM-L6-v2` from `sentence-transformers`).
            - Each chunk is enriched with metadata (type, name, parameters, decorators, docstring).
            - The model encodes this text into a vector (typically 384 dimensions).
            - Embeddings are cached and generated in batches for efficiency.

            ---
            **Analogy:**
            Imagine translating every paragraph in a book into a unique barcode that captures its meaning. Embeddings are like those barcodes for code!

            **Advanced Details:**
            - **Metadata fusion:** Embedding input includes not just code, but also type, name, parameters, and docstring for richer context.
            - **Batch processing:** Improves throughput and leverages model parallelism.
            - **Hash-based caching:** Avoids redundant computation for identical chunks.
            - **Supports model swapping:** Can use larger or domain-specific models for higher accuracy.

            **Why is this important?**
            - **Semantic Understanding:** Similar code (by meaning, not just words) gets similar vectors.
            - **Natural Language Support:** Enables searching code with plain English queries.
            - **Transfer Learning:** Leverages knowledge from massive code and text corpora.

            **Example:**
            ```python
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embedding = model.encode("Type: function Name: add Content: def add(a, b): ...")
            ```
            """)
            st.info("**Did you know?** Embeddings are the secret sauce behind AI search, recommendations, and even image recognition!")

            with st.expander("🧠 What is a Transformer Embedding?"):
                st.markdown("""
                - A **transformer** is a neural network architecture designed for understanding sequences (text, code).
                - **Embeddings** are dense vector representations capturing meaning and context.
                - Pre-trained models can generalize to new code and queries.

                **Technical Note:**
                Transformers use self-attention to model relationships between all tokens, making them ideal for code, which often has long-range dependencies.
                """)

            with st.expander("⚡ Performance Tips"):
                st.markdown("""
                - Embeddings are cached using a hash of the chunk text.
                - Batch processing (32 chunks at a time) speeds up embedding generation.
                - For very large codebases, embeddings can be precomputed and stored offline.
                """)

            with st.expander("🔍 Use Cases"):
                st.markdown("""
                - **Semantic code search:** Find code by intent, not just keywords.
                - **Duplicate detection:** Identify similar or copy-pasted code across repositories.
                - **Automated documentation:** Retrieve relevant code snippets for doc generation.
                """)

        with st.expander("3️⃣ Indexing & Storage - Creating a Searchable Code Library", expanded=False):
            st.markdown("""
            **Objective:**
            Store embeddings in a database that supports fast similarity search.

            **How it works:**
            - Uses **FAISS** (Facebook AI Similarity Search) for vector indexing.
            - Embeddings are normalized (L2) for cosine similarity.
            - Supports multiple index types:
                - `IndexFlatIP`: Exact search (cosine similarity)
                - `IndexPQ`: Quantized, for large datasets
                - (Optionally) HNSW for approximate search
            - All code chunks and their embeddings are stored for retrieval.

            ---
            **Analogy:**
            Think of FAISS as a super-powered librarian who can instantly find the most similar "barcodes" (embeddings) in a giant library!

            **Advanced Details:**
            - **Hybrid indexing:** Switches between exact and approximate search based on dataset size.
            - **Search statistics:** Tracks query frequency, average search time, and popular queries.
            - **Chunk metadata storage:** Each embedding is linked to its code chunk and metadata for rich result display.

            **Why is this important?**
            - **Speed:** Finds nearest neighbors in milliseconds, even for thousands of chunks.
            - **Scalability:** Handles large codebases efficiently.
            - **Flexibility:** Can switch between exact and approximate search.

            **Example:**
            ```python
            import faiss
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embedding_dim)
            index.add(embeddings.astype('float32'))
            ```
            """)
            st.info("**Tip:** Vector search is used by Google Photos, Spotify, and many AI assistants!")

            with st.expander("🗄️ What is FAISS and why use it?"):
                st.markdown("""
                - **FAISS** is a library for efficient similarity search and clustering of dense vectors.
                - Used by leading AI search and recommendation systems.
                - Supports CPU and GPU acceleration.
                - Can handle millions of vectors with sub-second latency.

                **Technical Note:**
                FAISS supports advanced indexing structures (IVF, PQ, HNSW) for balancing speed, memory, and accuracy.
                """)

            with st.expander("🔍 Use Cases"):
                st.markdown("""
                - **Instant code search:** Developers can find relevant code in real time.
                - **Similarity-based recommendations:** Suggest related functions or classes.
                - **Code deduplication:** Identify and merge similar code fragments.
                """)

        with st.expander("4️⃣ Retrieval & Search - Finding Relevant Code", expanded=False):
            st.markdown("""
            **Objective:**
            Retrieve the most relevant code chunks for a user's query (in natural language or code).

            **How it works:**
            1. User enters a query (e.g., "function that cleans data").
            2. The query is embedded using the same transformer model.
            3. The embedding is searched against the FAISS index.
            4. Top-k most similar code chunks are returned, ranked by similarity.
            5. Results can be filtered by type, complexity, or quality.

            ---
            **Analogy:**
            It's like asking a librarian a question and instantly getting the most relevant book paragraphs, even if you don't know the exact words!

            **Advanced Details:**
            - **Multi-modal queries:** Supports both code and natural language as input.
            - **Filtering and ranking:** Results can be filtered by chunk type, complexity, maintainability, or custom tags.
            - **Contextual expansion:** Can retrieve related chunks (e.g., methods in the same class, or functions called by the result).
            - **Search analytics:** Tracks query patterns to improve relevance over time.

            **Why is this important?**
            - **Semantic Search:** Finds code by meaning, not just keywords.
            - **Natural Language:** Users don't need to know exact function names.
            - **Contextual Awareness:** Handles synonyms, paraphrases, and related concepts.

            **Example:**
            ```python
            query_embedding = model.encode([query])[0]
            distances, indices = index.search(query_embedding, k)
            ```
            """)
            st.info("**Did you know?** This is how AI assistants can answer questions about your codebase!")

            with st.expander("🔍 Advanced: Filtering and Ranking"):
                st.markdown("""
                - Results can be filtered by:
                    - Chunk type (function, class, etc.)
                    - Maximum complexity
                    - Minimum quality score
                - Each result includes:
                    - Name, type, similarity score
                    - Complexity, maintainability, parameters, decorators

                **Technical Note:**
                Ranking can be further improved by combining semantic similarity with static code metrics or usage frequency.
                """)

            with st.expander("💡 Real-World Example"):
                st.markdown("""
                - **Find all data cleaning functions:** Query "clean missing values" and retrieve relevant utilities.
                - **Locate async handlers:** Filter results to only show async functions.
                """)

        with st.expander("5️⃣ Analytics & Insights - Understanding the Codebase", expanded=False):
            st.markdown("""
            **Objective:**
            Provide high-level analytics, clustering, and actionable insights about the codebase.

            **How it works:**
            - **Clustering:** Uses KMeans on embeddings to group related chunks.
            - **Dimensionality Reduction:** Uses PCA to visualize clusters in 2D.
            - **Complexity & Quality Metrics:** Calculates averages, distributions, and highlights outliers.
            - **Dependency Analysis:** Extracts call graphs, import graphs, and inheritance relationships.
            - **Recommendations:** Suggests improvements based on analysis (e.g., refactor complex code, add comments).

            ---
            **Analogy:**
            Imagine a map of your codebase, where similar code is grouped together, and hotspots are highlighted for improvement!

            **Advanced Details:**
            - **Semantic clustering:** Reveals hidden structure and architectural patterns in code.
            - **Hotspot detection:** Identifies complex or low-quality areas for targeted refactoring.
            - **Dependency visualization:** Graphs show how code components interact, aiding modularization.
            - **Continuous improvement:** Analytics can be tracked over time for CI/CD integration.

            **Why is this important?**
            - **Bird's-Eye View:** See structure, hotspots, and patterns at a glance.
            - **Quality Assurance:** Identify code that needs refactoring or documentation.
            - **Continuous Improvement:** Data-driven recommendations for maintainability.

            **Example:**
            ```python
            # KMeans clustering
            kmeans = KMeans(n_clusters=8)
            labels = kmeans.fit_predict(embeddings)
            # PCA for visualization
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)
            ```
            """)
            st.info("**Tip:** Visualizing your codebase helps spot hidden patterns and technical debt!")

            with st.expander("📈 Visualizations Provided"):
                st.markdown("""
                - **Complexity Distribution:** Bar chart of chunk complexity levels.
                - **Complexity Heatmap:** Visualizes complexity per chunk.
                - **Semantic Clustering:** 2D scatter plot of code clusters.
                - **Call Graph:** Network diagram of function/method calls.
                - **Inheritance Graph:** Shows class hierarchies and relationships.
                """)

            with st.expander("💡 Example Recommendations"):
                st.markdown("""
                - "Increase code comments and improve maintainability for better quality."
                - "Refactor highly complex functions/classes to reduce cyclomatic complexity."
                - "Consider using classes for better code organization and reusability."
                - "Review import statements for possible redundancy or unused imports."
                - "Modularize tightly coupled code to improve testability and reuse."
                """)

            with st.expander("🔬 Advanced: Integrating with DevOps"):
                st.markdown("""
                - **CI/CD integration:** Run analytics on every pull request to enforce quality gates.
                - **Trend analysis:** Track maintainability and complexity over time.
                - **Automated alerts:** Notify teams when code quality drops below thresholds.
                """)

        # --- NEW: LLM Demystified for Beginners ---
        with st.expander("🤖 BONUS: How Would You Build an LLM (Large Language Model) From Scratch?", expanded=False):
            st.markdown("""
            **Ever wondered how ChatGPT, Copilot, or Bard are built? Here's a beginner-friendly roadmap!**

            ---
            **Step 1: Gather Data - The Raw Material**
            - **What:** Collect a massive amount of text and code from the internet, books, open-source repositories, etc. Think petabytes (millions of gigabytes)!
            - **Why:** LLMs learn by reading patterns, grammar, facts, and coding styles from this data. The more diverse and high-quality the data, the better the model.
            - **Analogy:** This is like giving a student every book in the world to read.

            **Step 2: Tokenization - Breaking Down Language**
            - **What:** Convert the raw text/code into smaller units called "tokens". A token can be a word, part of a word, a punctuation mark, or a special symbol.
            - **Why:** Computers work with numbers, not text. Tokenization turns text into a sequence of numbers that the model can process. It also helps handle rare words or code patterns efficiently.
            - **Example:** The sentence `"Hello, world!"` might become `[1549, 11, 2150, 0]` where each number represents a token. The code `"def add(a, b):"` might become `[54, 102, 3, 8, 11, 8, 4, 2]` (simplified).
            - **Analogy:** Like breaking sentences into individual words or syllables before you learn to read.

            **Step 3: Model Architecture - The Brain**
            - **What:** Design the structure of the neural network. The most common architecture for LLMs is the **Transformer**.
            - **Why:** Transformers are great at understanding context and relationships between tokens, even if they are far apart in the sequence (like a variable defined at the top of a function and used at the bottom).
            - **Key Idea:** **Attention** mechanism - allows the model to focus on the most relevant parts of the input sequence when processing a token.
            - **Analogy:** Imagine having a super-powered memory that lets you instantly recall any relevant piece of information from everything you've ever read when trying to understand a new sentence.

            **Step 4: Training - The Learning Process**
            - **What:** Feed the tokenized data into the Transformer model and train it to predict the *next* token in a sequence, given the previous ones. This is called **autoregressive training**.
            - **Why:** By predicting the next token billions of times, the model learns grammar, facts, reasoning abilities, and coding patterns. It learns which tokens are likely to follow others in different contexts.
            - **Process:** The model makes a prediction, compares it to the actual next token, calculates an "error", and adjusts its internal parameters (weights and biases) to reduce that error. This is done using algorithms like **gradient descent**.
            - **Scale:** This step requires enormous computing power (thousands of GPUs) and takes weeks or months. The model has billions or even trillions of parameters (think of them as tiny knobs the model adjusts).
            - **Analogy:** Like practicing predicting the next word in a sentence over and over again, across the entire internet, until you become incredibly good at it.

            **Step 5: Fine-Tuning (Optional but Common)**
            - **What:** After the initial massive training (pre-training), the model can be trained further on a smaller, more specific dataset or for a specific task (like answering questions, writing code, or following instructions).
            - **Why:** This specializes the general-purpose pre-trained model for particular applications. For a code LLM, you'd fine-tune heavily on code-related tasks. For a chatbot, you'd fine-tune on conversational data.
            - **Techniques:** Includes Supervised Fine-Tuning (SFT) on input/output pairs and Reinforcement Learning from Human Feedback (RLHF) to align the model's output with human preferences.
            - **Analogy:** After graduating from the "internet university", you go to "code academy" or "chatbot finishing school" to become an expert in a specific area.

            **Step 6: Inference & Serving - Putting the Model to Work**
            - **What:** Deploy the trained and potentially fine-tuned model so users can interact with it. When a user provides a "prompt" (input text/code), the model generates a response one token at a time, based on what it learned during training.
            - **Why:** This is the step where the model actually *does* something useful!
            - **Challenges:** Running large models efficiently requires specialized hardware and software for fast response times and managing many users.
            - **Analogy:** Opening the library to the public and having the super-powered librarian answer questions based on everything they've read.

            ---
            **Analogy Summary:**
            - **Data:** The entire library of human knowledge (text & code).
            - **Tokenization:** Breaking books into words/syllables.
            - **Model Architecture (Transformer):** The librarian's brain with incredible memory (attention).
            - **Training:** The librarian reading *everything* and practicing predicting the next word.
            - **Fine-Tuning:** The librarian specializing in a specific topic (like coding).
            - **Inference:** The librarian answering your questions instantly.

            ---
            **Did you know?**
            - Modern LLMs have billions of parameters (think: adjustable dials).
            - Open-source LLMs like Llama, Mistral, and StarCoder are available for anyone to experiment with.
            - You can build a *tiny* LLM on your laptop using libraries like `transformers` and a small dataset!
            - The cost of training a large LLM can be millions of dollars!

            ---
            **Want to try?**
            - Check out [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for tutorials on using pre-trained models.
            - Try training a mini language model on your own code snippets!

            ---
            **Summary Table: LLM Building Blocks**

            | Step           | What Happens?                              | Analogy                                  | Tools/Libraries         |
            |----------------|--------------------------------------------|------------------------------------------|-------------------------|
            | Data           | Collect text/code                          | The Library                              | GitHub, Common Crawl    |
            | Tokenization   | Split into tokens                          | Breaking text into words/syllables       | `tokenizers`, `sentencepiece` |
            | Model          | Build transformer architecture             | The Librarian's Brain (with Attention)   | `transformers`, `pytorch`, `tensorflow` |
            | Training       | Learn to predict next token                | Reading & Practicing Predictions         | `accelerate`, `deepspeed` |
            | Fine-tuning    | Specialize for tasks                       | Specializing in a Topic                  | `peft`, `trl`           |
            | Inference      | Serve model for user queries               | Answering Questions for the Public       | `fastapi`, `vllm`, `onnx` |

            ---
            **You now know the high-level steps to build your own LLM!**
            """)

if __name__ == "__main__":
    main()
