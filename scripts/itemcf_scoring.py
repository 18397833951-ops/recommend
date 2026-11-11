"""评分版 ItemCF 的逐用户预测，供驱动进程与 Spark 工作进程共用。"""

import math


def predict_user(user_id, history, targets, user_mean, catalog, global_mean, model):
    """catalog: movieId -> (电影训练均值, 邻居 ID 数组, 对应相似度数组)。"""
    known_user = history is not None
    ratings = dict(history) if known_user else {}
    for movie_id, rating, timestamp, source in targets:
        item = catalog.get(movie_id)
        known_movie = item is not None
        neighbors, numerator, denominator = 0, 0.0, 0.0
        if known_movie:
            base, ids, similarities = item
            for neighbor, similarity in zip(ids, similarities):
                residual = ratings.get(int(neighbor))
                if residual is not None:
                    neighbors += 1
                    numerator += float(similarity) * residual
                    denominator += float(similarity)
        else:
            base = user_mean if known_user else global_mean
        fallback = neighbors < model["min_neighbors"] or denominator <= 0
        raw = base if fallback else base + numerator / denominator
        if not math.isfinite(raw):
            raise ValueError("ItemCF 预测不是有限数值。")
        prediction = min(model["rating_max"], max(model["rating_min"], raw)) if model["clip_predictions"] else raw
        yield (int(user_id), int(movie_id), float(rating), int(timestamp), str(source), float(raw),
               float(prediction), neighbors, not known_user, not known_movie, fallback)
