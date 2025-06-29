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
    st.sidebar.header("Upload & Settings")

    uploaded_file = st.sidebar.file_uploader("Upload Python file", type=["py"])
    example_code = st.sidebar.checkbox("Use example code", value=False)
    search_query = st.sidebar.text_input("Semantic Search Query", "")
    k_results = st.sidebar.slider("Top-K Results", 1, 10, 5)
    show_analytics = st.sidebar.checkbox("Show Analytics", value=True)

    if uploaded_file or example_code:
        if uploaded_file:
            code = uploaded_file.read().decode("utf-8")
            file_path = uploaded_file.name
        else:
            code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

class Calculator:
    def multiply(self, x, y):
        # Multiplies two numbers
        return x * y
            '''
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

            st.markdown("#### Quality Metrics")
            st.json(analysis["quality_metrics"])

            st.markdown("#### Structure Analysis")
            st.json(analysis["structure_analysis"])

            st.markdown("#### Recommendations")
            for rec in analysis["recommendations"]:
                st.info(rec)

            if analysis["semantic_clustering"]:
                st.markdown("#### Semantic Clustering (2D PCA)")
                clusters = pd.DataFrame(analysis["semantic_clustering"]["clusters"])
                fig = px.scatter(
                    clusters, x="x", y="y", color="cluster", hover_data=["name", "type"],
                    title="Semantic Clusters of Code Chunks"
                )
                st.plotly_chart(fig, use_container_width=True)

        if search_query:
            st.subheader(f"🔎 Semantic Search Results for: '{search_query}'")
            results = rag.search_code(search_query, k=k_results)
            for res in results:
                chunk = res["chunk"]
                st.markdown(
                    f"<div class='metric-card'><b>{chunk.name}</b> "
                    f"({chunk.chunk_type}, Complexity: <span class='complexity-indicator complexity-{chunk.complexity_category.lower()}'>{chunk.complexity_category}</span>)<br>"
                    f"<pre>{chunk.content}</pre>"
                    f"<small>Similarity: {res['similarity']:.3f} | Quality: {res['quality_score']:.2f} | Lines: {res['lines']}</small></div>",
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main()
