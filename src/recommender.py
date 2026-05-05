from dataclasses import dataclass
import random
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class GAConfig:
    rating_w: float = 0.35
    behavior_w: float = 0.25
    category_w: float = 0.15
    price_w: float = 0.10
    diversity_w: float = 0.10
    novelty_w: float = 0.05


class GARecommender:
    """Genetic Algorithm recommender for the BIA601 assignment."""

    def __init__(self, users: pd.DataFrame, products: pd.DataFrame,
                 ratings: pd.DataFrame, behavior: pd.DataFrame, seed: int = 42):
        self.users = users.copy()
        self.products = products.copy()
        self.ratings = ratings.copy()
        self.behavior = behavior.copy()
        self.config = GAConfig()
        random.seed(seed)
        np.random.seed(seed)
        self._prepare_scores()

    def _prepare_scores(self):
        product_rating = (
            self.ratings.groupby("product_id")["rating"]
            .mean()
            .rename("avg_rating")
            .reset_index()
        )
        product_behavior = (
            self.behavior.groupby("product_id")[["viewed", "clicked", "purchased"]]
            .mean()
            .reset_index()
        )

        self.product_table = self.products.merge(
            product_rating, left_on="id_product", right_on="product_id", how="left"
        ).drop(columns=["product_id"], errors="ignore")

        self.product_table = self.product_table.merge(
            product_behavior, left_on="id_product", right_on="product_id", how="left"
        ).drop(columns=["product_id"], errors="ignore")

        self.product_table["avg_rating"] = self.product_table["avg_rating"].fillna(3.0)
        for col in ["viewed", "clicked", "purchased"]:
            self.product_table[col] = self.product_table[col].fillna(0.0)

        max_price = max(float(self.product_table["price"].max()), 1.0)
        self.product_table["rating_score"] = self.product_table["avg_rating"] / 5.0
        self.product_table["behavior_score"] = (
            0.20 * self.product_table["viewed"]
            + 0.30 * self.product_table["clicked"]
            + 0.50 * self.product_table["purchased"]
        )
        popularity = self.behavior.groupby("product_id").size().rename("popularity").reset_index()
        self.product_table = self.product_table.merge(
            popularity, left_on="id_product", right_on="product_id", how="left"
        ).drop(columns=["product_id"], errors="ignore")
        self.product_table["popularity"] = self.product_table["popularity"].fillna(0)
        max_pop = max(float(self.product_table["popularity"].max()), 1.0)
        self.product_table["novelty_score"] = 1.0 - (self.product_table["popularity"] / max_pop)
        self.max_price = max_price

    def _user_profile(self, user_id):
        user_ratings = self.ratings[self.ratings["user_id"] == user_id]
        user_behavior = self.behavior[self.behavior["user_id"] == user_id]

        interacted_products = pd.concat([
            user_ratings[["product_id"]],
            user_behavior[["product_id"]]
        ], ignore_index=True).drop_duplicates()

        interacted = interacted_products.merge(
            self.products, left_on="product_id", right_on="id_product", how="left"
        )

        if interacted.empty:
            category_weights = {}
            avg_price = float(self.products["price"].median())
            seen_ids = set()
        else:
            category_weights = (
                interacted["category"].value_counts(normalize=True).to_dict()
            )
            avg_price = float(interacted["price"].fillna(self.products["price"].median()).mean())
            seen_ids = set(interacted["product_id"].tolist())

        return category_weights, avg_price, seen_ids

    def _candidate_table(self, user_id):
        category_weights, avg_price, seen_ids = self._user_profile(user_id)
        candidates = self.product_table[~self.product_table["id_product"].isin(seen_ids)].copy()
        if candidates.empty:
            candidates = self.product_table.copy()

        candidates["category_score"] = candidates["category"].map(category_weights).fillna(0.05)
        candidates["price_score"] = 1.0 - (abs(candidates["price"] - avg_price) / self.max_price)
        candidates["price_score"] = candidates["price_score"].clip(0, 1)

        cfg = self.config
        candidates["base_score"] = (
            cfg.rating_w * candidates["rating_score"]
            + cfg.behavior_w * candidates["behavior_score"]
            + cfg.category_w * candidates["category_score"]
            + cfg.price_w * candidates["price_score"]
            + cfg.novelty_w * candidates["novelty_score"]
        )

        return candidates.sort_values("base_score", ascending=False)

    def _chromosome_fitness(self, chrom: List, candidates: pd.DataFrame) -> float:
        sub = candidates[candidates["id_product"].isin(chrom)]
        if sub.empty:
            return 0.0
        base = float(sub["base_score"].mean())
        diversity = sub["category"].nunique() / max(len(chrom), 1)
        return base + self.config.diversity_w * diversity

    def _make_chromosome(self, ids: List, k: int) -> List:
        if len(ids) <= k:
            return ids.copy()
        return random.sample(ids, k)

    def _tournament(self, population: List[List], fitnesses: List[float]) -> List:
        group = random.sample(list(zip(population, fitnesses)), k=min(3, len(population)))
        return max(group, key=lambda x: x[1])[0].copy()

    def _crossover(self, p1: List, p2: List, ids: List, k: int) -> List:
        if k <= 1:
            return p1.copy()
        point = random.randint(1, k - 1)
        child = p1[:point] + [x for x in p2 if x not in p1[:point]]
        while len(child) < k:
            new_id = random.choice(ids)
            if new_id not in child:
                child.append(new_id)
        return child[:k]

    def _mutate(self, chrom: List, ids: List, mutation_rate: float) -> List:
        child = chrom.copy()
        for i in range(len(child)):
            if random.random() < mutation_rate:
                alternatives = [x for x in ids if x not in child]
                if alternatives:
                    child[i] = random.choice(alternatives)
        return child

    def recommend(self, user_id, k: int = 10, population_size: int = 60,
                  generations: int = 60, mutation_rate: float = 0.12) -> Tuple[pd.DataFrame, list]:
        candidates = self._candidate_table(user_id)
        ids = candidates["id_product"].tolist()
        if not ids:
            return pd.DataFrame(), []

        k = min(k, len(ids))
        population = [self._make_chromosome(ids, k) for _ in range(population_size)]
        history = []

        best_chrom = population[0]
        best_fit = -1.0

        for gen in range(generations):
            fitnesses = [self._chromosome_fitness(ch, candidates) for ch in population]
            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_fit:
                best_fit = fitnesses[gen_best_idx]
                best_chrom = population[gen_best_idx].copy()

            history.append((gen, best_fit))
            new_population = [best_chrom.copy()]  # elitism

            while len(new_population) < population_size:
                parent1 = self._tournament(population, fitnesses)
                parent2 = self._tournament(population, fitnesses)
                child = self._crossover(parent1, parent2, ids, k)
                child = self._mutate(child, ids, mutation_rate)
                new_population.append(child)

            population = new_population

        result = candidates[candidates["id_product"].isin(best_chrom)].copy()
        result["final_fitness"] = result["id_product"].map({
            pid: candidates.loc[candidates["id_product"] == pid, "base_score"].iloc[0]
            for pid in best_chrom
        })
        result = result[[
            "id_product", "category", "price", "avg_rating",
            "behavior_score", "category_score", "price_score",
            "novelty_score", "final_fitness"
        ]].sort_values("final_fitness", ascending=False)

        result["reason"] = result.apply(self._explain, axis=1)
        return result, history

    @staticmethod
    def _explain(row) -> str:
        reasons = []
        if row["avg_rating"] >= 4:
            reasons.append("high rating")
        if row["behavior_score"] >= 0.5:
            reasons.append("strong behavior signals")
        if row["category_score"] >= 0.2:
            reasons.append("matches user categories")
        if row["price_score"] >= 0.7:
            reasons.append("suitable price range")
        if not reasons:
            reasons.append("balanced recommendation")
        return ", ".join(reasons)
