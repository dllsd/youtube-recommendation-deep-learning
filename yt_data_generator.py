import numpy as np
import pandas as pd

np.random.seed(42)

# Parameters
num_users = 500
num_videos = 1000
num_samples = 10000
num_categories = 5

# Generate users
users = pd.DataFrame({
    "user_id": np.arange(num_users),
    "user_avg_watch_ratio": np.random.uniform(0.4, 1.0, num_users),
    "user_avg_video_length": np.random.uniform(5, 30, num_users),
    "user_activity_level": np.random.uniform(0.2, 1.0, num_users),
    "user_preferred_category": np.random.randint(0, num_categories, num_users)
})

# Generate videos
videos = pd.DataFrame({
    "video_id": np.arange(num_videos),
    "video_length": np.random.uniform(5, 30, num_videos),
    "video_popularity": np.random.uniform(0, 1, num_videos),
    "video_recency": np.random.uniform(0, 1, num_videos),
    "video_category": np.random.randint(0, num_categories, num_videos)
})

# Generate recommendation samples
data = []

for _ in range(num_samples):
    user = users.sample(1).iloc[0]
    video = videos.sample(1).iloc[0]

    category_match = int(user["user_preferred_category"] == video["video_category"])
    length_similarity = 1 - abs(user["user_avg_video_length"] - video["video_length"]) / 25
    length_similarity = max(0, length_similarity)

    score = (
        1.5 * category_match +
        1.2 * length_similarity +
        1.0 * video["video_popularity"] +
        0.8 * user["user_activity_level"] +
        0.5 * video["video_recency"]
    )

    probability = 1 / (1 + np.exp(-(score - 2.5)))
    clicked = np.random.binomial(1, probability)

    data.append([
        user["user_avg_watch_ratio"],
        user["user_avg_video_length"],
        user["user_activity_level"],
        video["video_length"],
        video["video_popularity"],
        video["video_recency"],
        category_match,
        length_similarity,
        clicked
    ])

columns = [
    "user_avg_watch_ratio",
    "user_avg_video_length",
    "user_activity_level",
    "video_length",
    "video_popularity",
    "video_recency",
    "category_match",
    "length_similarity",
    "clicked"
]

dataset = pd.DataFrame(data, columns=columns)
dataset.to_csv("youtube_recommendation_dataset.csv", index=False)

print("Dataset created:", dataset.shape)