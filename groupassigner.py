import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss


class SemanticGroupAssigner:
    """
    Manager that:
    - Builds embeddings from product records (SBERT)
    - Maintains group centroids and FAISS index for centroids (IndexFlatIP)
    - Assigns new products to existing group if cosine similarity >= threshold,
      otherwise creates new group.
    - Supports buffering and flushing for batch updates.
    """

    def __init__(self,
                 embed_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 similarity_threshold: float = 0.80,
                 buffer_flush: int = 50,
                 embedding_batch_size: int = 64):
        self.embed_model_name = embed_model_name
        self.similarity_threshold = similarity_threshold
        self.buffer_flush = buffer_flush
        self.embedding_batch_size = embedding_batch_size

        # runtime items
        self.model: Optional[SentenceTransformer] = None
        self.df: Optional[pd.DataFrame] = None  # stores product records including group_sku_id
        self.product_embeddings: Optional[np.ndarray] = None  # float32 (N, D)

        # groups: lists
        self.group_centroids: List[np.ndarray] = []  # normalized float32 centroids (G, D)
        self.group_member_counts: List[int] = []     # number of members per group
        self.group_index: Optional[faiss.IndexFlatIP] = None  # NN index over centroids

        # buffer for incremental adds (series, vector)
        self._buffer: List[Tuple[Dict[str, Any], np.ndarray]] = []

        self._load_model()

    # ---------------- model ----------------
    def _load_model(self):
        print(f"Loading SBERT model: {self.embed_model_name} ...")
        self.model = SentenceTransformer(self.embed_model_name)
        print("Model loaded.")

    # ---------------- helpers ----------------
    def _read_json(self, json_file: str) -> dict:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _flatten_attributes(self, attr_products: Optional[dict]) -> dict:
        if not attr_products or 'data' not in attr_products:
            return {}
        result = {}
        for a in attr_products.get('data', []):
            key = a.get('attributes', {}).get('name') or f"attr_{a.get('attribute_id')}"
            value = a.get('attribute_values', {}).get('name') if a.get('attribute_values') else None
            if key and value:
                result[key] = value
        return result

    def _clean_text(self, text: str) -> str:
        if text is None:
            return ""
        text = text.lower()
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"\[.*?\]", " ", text) # Remove text in square brackets
        text = re.sub(r"[^\w\s]", " ", text) # remove punctuation
        text = re.sub(r"\s+", " ", text).strip() # remove extra whitespace
        return text

    def _transform_data_from_json(self, json_file: str) -> pd.DataFrame:
        products = self._read_json(json_file)['data']
        result_list = []

        for product in products:
            skus_data = product.get('product_skus', {}).get('data', [])[0]
            if not skus_data:
                continue
            product_info = {
                "product_id": product.get("id"),
                "product_name": product.get("name"),
                "brand_name": product.get("brand", {}).get("name"),
                "short_description": product.get("short_description"),
                "seller_sku": skus_data.get("seller_sku"),
                "attribute_products": {
                    "data": self._flatten_attributes(product.get("attribute_products"))
                },
            }
            result_list.append(product_info)
        return pd.DataFrame(result_list)

    def _get_text_for_embedding(self, product_row: pd.Series) -> str:
        name = self._clean_text(product_row.get('product_name', ''))
        brand = self._clean_text(product_row.get('brand_name', ''))
        desc = self._clean_text(product_row.get('short_description', ''))
        seller_sku = self._clean_text(str(product_row.get('seller_sku', '')))

        keyword_set = set()
        if name: keyword_set.add(name)
        if brand: keyword_set.add(brand)
        if seller_sku: keyword_set.add(seller_sku)
        attributes_data = product_row.get('attribute_products', {}).get('data', {})
        if isinstance(attributes_data, dict):
            for key, value in attributes_data.items():
                if value:
                    keyword_set.add(self._clean_text(str(value)))

        keyword_text = " | ".join(sorted(list(keyword_set)))
        return f"""Product: {name}
Brand: {brand}
Seller SKU code: {seller_sku}
Description: {desc}
Keywords: {keyword_text}"""

    # ---------------- building from JSON ----------------
    def build_from_json(self, json_file: str):
        """
        Build initial DataFrame and groups using greedy centroid assignment:
        iterate products in file order; for each product, find nearest existing
        centroid and assign if similarity >= threshold else create new group.
        """
        print(f"Loading products from {json_file} ...")
        self.df = self._transform_data_from_json(json_file)
        
        # compute embeddings in batches
        print("Computing embeddings for all products...")
        texts = self.df.apply(self._get_text_for_embedding, axis=1).tolist()
        embeddings = self.model.encode(texts, batch_size=self.embedding_batch_size, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype='float32')
        faiss.normalize_L2(embeddings)
        self.product_embeddings = embeddings

        # Greedy grouping using centroids
        print("Greedy grouping into centroids (no clustering lib)...")
        D = embeddings.shape[1]
        self.group_centroids = []
        self.group_member_counts = []
        assigned_group_ids = np.full(len(embeddings), -1, dtype=int)

        for i, vec in enumerate(embeddings):
            if len(self.group_centroids) == 0:
                # create first group
                self.group_centroids.append(vec.copy())
                self.group_member_counts.append(1)
                assigned_group_ids[i] = 0
            else:
                # search against centroid index
                self._build_centroid_index()  # ensures index exists and is up-to-date
                vec_q = vec.reshape(1, -1)
                Dsim, Is = self.group_index.search(vec_q, 1)
                sim = float(Dsim[0][0])
                nearest_gid = int(Is[0][0])
                if sim >= self.similarity_threshold:
                    # assign to that group; update centroid = (c*n + vec) / (n+1)
                    n = self.group_member_counts[nearest_gid]
                    c = self.group_centroids[nearest_gid]
                    new_c = (c * n + vec) / (n + 1)
                    faiss.normalize_L2(new_c.reshape(1, -1))
                    self.group_centroids[nearest_gid] = new_c
                    self.group_member_counts[nearest_gid] += 1
                    assigned_group_ids[i] = nearest_gid
                    # rebuild index next iteration will reflect centroid change
                else:
                    # create new group
                    new_gid = len(self.group_centroids)
                    self.group_centroids.append(vec.copy())
                    self.group_member_counts.append(1)
                    assigned_group_ids[i] = new_gid

        self.df['group_sku_id'] = assigned_group_ids
        # final build of centroid index
        self._build_centroid_index()
        print(f"Built {len(self.group_centroids)} groups from {len(self.df)} products.")
        # optionally save results
        self.df.to_csv('grouped_products.csv', index=False)

    def _build_centroid_index(self):
        """
        (Re)build FAISS index for current centroids.
        Fast for moderate number of groups (G). Rebuilds whenever centroids change.
        """
        if len(self.group_centroids) == 0:
            self.group_index = None
            return
        centroids = np.vstack(self.group_centroids).astype('float32')
        faiss.normalize_L2(centroids)
        d = centroids.shape[1]
        # IndexFlatIP is exact inner-product search; vectors are normalized so IP ~ cosine
        self.group_index = faiss.IndexFlatIP(d)
        self.group_index.add(centroids)

    # ---------------- assigning new product (real-time) ----------------
    def assign_new_product(self, product_dict: Dict[str, Any]) -> int:
        """
        Assign a group to a new product (real-time). Adds to buffer.
        Returns assigned group id (int). -1 if error.
        """
        if self.model is None or self.df is None:
            raise Exception("Not initialized. Run build_from_json() first.")

        text = self._get_text_for_embedding(product_dict)
        vec = self.model.encode([text]).astype('float32')
        faiss.normalize_L2(vec)

        # if no groups exist yet, create first group
        if self.group_index is None or len(self.group_centroids) == 0:
            gid = 0
            self.group_centroids.append(vec[0].copy())
            self.group_member_counts.append(1)
            self._build_centroid_index()
            assigned_gid = gid
        else:
            Dsim, Is = self.group_index.search(vec, 1)
            sim = float(Dsim[0][0])
            nearest_gid = int(Is[0][0])
            if sim >= self.similarity_threshold:
                # assign and update centroid (we'll update centroid only upon flush to reduce churn or optionally update immediately)
                # For accuracy update immediately:
                n = self.group_member_counts[nearest_gid]
                c = self.group_centroids[nearest_gid]
                new_c = (c * n + vec[0]) / (n + 1)
                faiss.normalize_L2(new_c.reshape(1, -1))
                self.group_centroids[nearest_gid] = new_c
                self.group_member_counts[nearest_gid] += 1
                self._build_centroid_index()
                assigned_gid = nearest_gid
            else:
                # create new group
                new_gid = len(self.group_centroids)
                self.group_centroids.append(vec[0].copy())
                self.group_member_counts.append(1)
                self._build_centroid_index()
                assigned_gid = new_gid

        # append to buffer for eventual flush to main df/embeddings
        product_record = dict(product_dict)  # shallow copy
        product_record['group_sku_id'] = int(assigned_gid)
        self._buffer.append((product_record, vec[0].copy()))

        if len(self._buffer) >= self.buffer_flush:
            self.flush_buffer()

        return int(assigned_gid)

    def flush_buffer(self):
        """
        Persist buffered products into self.df and self.product_embeddings.
        """
        if not self._buffer:
            print("Buffer empty, nothing to flush.")
            return
        recs = [r for r, v in self._buffer]
        vecs = np.vstack([v for r, v in self._buffer]).astype('float32')
        faiss.normalize_L2(vecs)

        # append to df
        df_new = pd.DataFrame(recs)
        self.df = pd.concat([self.df, df_new], ignore_index=True)

        # append to product_embeddings
        if self.product_embeddings is None:
            self.product_embeddings = vecs
        else:
            self.product_embeddings = np.vstack([self.product_embeddings, vecs])

        print(f"Flushed {len(self._buffer)} products to main store.")
        self._buffer = []

    # ---------------- save / load state ----------------
    def save_state(self, path_dir: str = "."):
        """Save centroid vectors, counts and df to disk."""
        os.makedirs(path_dir, exist_ok=True)
        # save df
        if self.df is not None:
            self.df.to_csv(os.path.join(path_dir, "products_with_groups.csv"), index=False)
        # save centroids
        if self.group_centroids:
            cent_arr = np.vstack(self.group_centroids).astype('float32')
            np.save(os.path.join(path_dir, "group_centroids.npy"), cent_arr)
            np.save(os.path.join(path_dir, "group_counts.npy"), np.array(self.group_member_counts, dtype=int))
        print(f"Saved state to {path_dir}")

    def load_state(self, path_dir: str = "."):
        """Load state if exists (centroids and df)."""
        df_path = os.path.join(path_dir, "products_with_groups.csv")
        cent_path = os.path.join(path_dir, "group_centroids.npy")
        count_path = os.path.join(path_dir, "group_counts.npy")

        if os.path.exists(df_path):
            self.df = pd.read_csv(df_path)
        if os.path.exists(cent_path) and os.path.exists(count_path):
            cent = np.load(cent_path)
            counts = np.load(count_path).tolist()
            self.group_centroids = [cent[i].astype('float32') for i in range(cent.shape[0])]
            self.group_member_counts = counts
            self._build_centroid_index()
        print(f"Loaded state from {path_dir}")
