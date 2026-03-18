"""
LLM Agent Movie Recommendation System (MovieTweetings)
Dual version: User side (new/returning user) + Admin monitoring system (overview + user details)
Features:
- User side: new users directly enter cold start + AI assistant; returning users login to personalized + franchise + AI assistant + cold start
- Admin side: overall statistics charts + text analysis + detailed analysis by user ID
Author: Customized according to project requirements
Date: 2026-03-07
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from collections import Counter
from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import faiss
from sentence_transformers import SentenceTransformer
import openai
import warnings

warnings.filterwarnings('ignore')

# ----------------------------- Configuration -----------------------------
DATA_PATH = "/Users/qxjq/PyCharmMiscProject/547 LLM/MovieTweetings-master"
RATINGS_FILE = os.path.join(DATA_PATH, "latest", "ratings.dat")
MOVIES_FILE = os.path.join(DATA_PATH, "latest", "movies.dat")

USE_REAL_LLM = False
if USE_REAL_LLM:
    openai.api_key = os.getenv("OPENAI_API_KEY")


# ----------------------------- Data Loading & Preprocessing -----------------------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_data
def load_ratings():
    cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    df = pd.read_csv(RATINGS_FILE, sep='::', engine='python', names=cols, encoding='latin-1')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df


@st.cache_data
def load_movies():
    cols = ['movie_id', 'title', 'genres']
    df = pd.read_csv(MOVIES_FILE, sep='::', engine='python', names=cols, encoding='latin-1')
    years = []
    clean_titles = []
    for t in df['title']:
        match = re.search(r'\((\d{4})\)$', t)
        if match:
            years.append(int(match.group(1)))
            clean_titles.append(t.replace(f"({match.group(1)})", "").strip())
        else:
            years.append(0)
            clean_titles.append(t)
    df['year'] = years
    df['clean_title'] = clean_titles
    df['genres'] = df['genres'].fillna('')
    df['genres_list'] = df['genres'].apply(lambda x: x.split('|') if x else [])
    return df


def build_movie_texts(movies_df):
    texts = []
    for _, row in movies_df.iterrows():
        genres_str = ', '.join(row['genres_list'])
        text = f"{row['clean_title']} is a movie in genres: {genres_str}. It was released in {row['year']}."
        texts.append(text)
    return texts


@st.cache_resource
def build_faiss_index(_movies_df, _embedder):
    texts = build_movie_texts(_movies_df)
    embeddings = _embedder.encode(texts, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index, embeddings, texts


# ----------------------------- LLM Agent Class -----------------------------
class MovieAgent:
    def __init__(self, movies_df, ratings_df, embedder, faiss_index):
        self.movies = movies_df
        self.ratings = ratings_df
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.popular_movies = self._get_popular_movies(50)
        self.latest_movies = self.movies[self.movies['year'] > 0].sort_values('year', ascending=False).head(50)
        self.all_tags = sorted(set([tag for sublist in movies_df['genres_list'] for tag in sublist if tag]))

    def _get_popular_movies(self, top_n=50):
        rating_counts = self.ratings.groupby('movie_id').size().reset_index(name='count')
        top_movies = rating_counts.sort_values('count', ascending=False).head(top_n)
        return self.movies[self.movies['movie_id'].isin(top_movies['movie_id'])]

    def cold_start_recommend(self, rec_type="both", top_k=10):
        if rec_type == "latest":
            recs = self.latest_movies.head(top_k)
            reason = "These are the latest movies in our database."
        elif rec_type == "popular":
            recs = self.popular_movies.head(top_k)
            reason = "These are the most popular movies based on user ratings."
        else:
            latest = self.latest_movies.head(top_k // 2)
            popular = self.popular_movies.head(top_k // 2)
            recs = pd.concat([latest, popular]).drop_duplicates(subset=['movie_id']).head(top_k)
            reason = "Here is a mix of the latest and most popular movies."
        return recs[['movie_id', 'clean_title', 'year', 'genres']], reason

    def recommend_by_tags(self, selected_tags, top_k=10):
        if not selected_tags:
            return pd.DataFrame(), "No tags selected."
        mask = self.movies['genres_list'].apply(lambda x: any(tag in x for tag in selected_tags))
        candidate_movies = self.movies[mask].copy()
        if candidate_movies.empty:
            return pd.DataFrame(), "No movies found with the selected tags."
        rating_counts = self.ratings.groupby('movie_id').size().reset_index(name='popularity')
        candidate_movies = candidate_movies.merge(rating_counts, on='movie_id', how='left').fillna(0)
        candidate_movies = candidate_movies.sort_values('popularity', ascending=False).head(top_k)
        reason = f"Based on your selected tags: {', '.join(selected_tags)}."
        return candidate_movies[['movie_id', 'clean_title', 'year', 'genres']], reason

    def recommend_by_query(self, query, top_k=10):
        if not query or not query.strip():
            return pd.DataFrame(), "Empty query."
        years_found = re.findall(r'\b(19|20)\d{2}\b', query)
        query_lower = query.lower()
        tags_found = [tag for tag in self.all_tags if tag.lower() in query_lower]

        mask = pd.Series([True] * len(self.movies))
        if years_found:
            year = int(years_found[0])
            mask &= (self.movies['year'] == year)
        if tags_found:
            mask &= self.movies['genres_list'].apply(lambda x: any(tag in x for tag in tags_found))
        elif not years_found:
            return self.cold_start_recommend("popular", top_k)

        candidate_movies = self.movies[mask].copy()
        if candidate_movies.empty:
            return pd.DataFrame(), "No movies found matching your query."

        rating_counts = self.ratings.groupby('movie_id').size().reset_index(name='popularity')
        candidate_movies = candidate_movies.merge(rating_counts, on='movie_id', how='left').fillna(0)
        candidate_movies = candidate_movies.sort_values('popularity', ascending=False).head(top_k)

        reason_parts = []
        if years_found:
            reason_parts.append(f"year {years_found[0]}")
        if tags_found:
            reason_parts.append(f"tags: {', '.join(tags_found)}")
        reason = "Based on your query: " + ", ".join(reason_parts) if reason_parts else "Based on your query."
        return candidate_movies[['movie_id', 'clean_title', 'year', 'genres']], reason

    def generate_user_profile(self, user_id, top_n=10):
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        if user_ratings.empty:
            return None, []
        liked = user_ratings[user_ratings['rating'] >= 7].sort_values('rating', ascending=False)
        if len(liked) < 3:
            liked = user_ratings.sort_values('rating', ascending=False).head(5)
        liked_movies = self.movies[self.movies['movie_id'].isin(liked['movie_id'])]
        if liked_movies.empty:
            return None, []

        movie_list = "\n".join([f"- {row['clean_title']} ({row['year']}) [{row['genres']}]"
                                for _, row in liked_movies.iterrows()])
        prompt = f"""Based on the following movies that a user liked, please summarize the user's movie preferences in 3-5 sentences. 
Focus on genres, themes, era, or any patterns you observe.
Movies liked:
{movie_list}
User preference summary:"""

        if USE_REAL_LLM:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=150
                )
                profile = response.choices[0].message.content.strip()
            except Exception as e:
                profile = f"[LLM call failed, using rule-based] User seems to enjoy movies similar to: {', '.join(liked_movies['clean_title'].head(3).tolist())}."
        else:
            all_genres = []
            for gl in liked_movies['genres_list']:
                all_genres.extend(gl)
            top_genres = Counter(all_genres).most_common(3)
            genre_str = ', '.join([g for g, _ in top_genres])
            years = liked_movies['year'].tolist()
            avg_year = int(np.mean([y for y in years if y > 0])) if any(y > 0 for y in years) else "various"
            profile = f"This user prefers {genre_str} movies, often from around {avg_year}. They enjoy titles like {liked_movies['clean_title'].iloc[0]} and {liked_movies['clean_title'].iloc[1] if len(liked_movies) > 1 else ''}."
        return profile, liked_movies

    def personalized_recommend(self, user_id, top_k=10):
        profile, liked_movies = self.generate_user_profile(user_id)
        if profile is None:
            return self.cold_start_recommend("popular", top_k)

        profile_emb = self.embedder.encode([profile]).astype('float32')
        distances, indices = self.faiss_index.search(profile_emb, top_k * 3)
        rec_movies = self.movies.iloc[indices[0]].copy()
        seen_ids = set(self.ratings[self.ratings['user_id'] == user_id]['movie_id'])
        rec_movies = rec_movies[~rec_movies['movie_id'].isin(seen_ids)]
        rec_movies = rec_movies.head(top_k)
        if len(rec_movies) < top_k:
            extra = self.popular_movies[~self.popular_movies['movie_id'].isin(seen_ids)].head(top_k - len(rec_movies))
            rec_movies = pd.concat([rec_movies, extra])
        return rec_movies[['movie_id', 'clean_title', 'year', 'genres']], profile

    def franchise_recommend(self, movie_id, top_k=5):
        movie_row = self.movies[self.movies['movie_id'] == movie_id]
        if movie_row.empty:
            return pd.DataFrame(), "No franchise found."
        title = movie_row.iloc[0]['clean_title']
        words = title.split()
        if len(words) >= 2:
            prefix = ' '.join(words[:2])
        else:
            prefix = title
        franchise = self.movies[self.movies['clean_title'].str.contains(prefix, case=False, na=False)]
        franchise = franchise[franchise['movie_id'] != movie_id].head(top_k)
        if franchise.empty:
            genres = movie_row.iloc[0]['genres_list']
            similar_genre = self.movies[self.movies['genres'].apply(lambda x: any(g in x for g in genres))]
            similar_genre = similar_genre[similar_genre['movie_id'] != movie_id].head(top_k)
            return similar_genre[['movie_id', 'clean_title', 'year', 'genres']], "Based on similar genres."
        reason = f"Other movies in the same franchise as '{title}'."
        return franchise[['movie_id', 'clean_title', 'year', 'genres']], reason


# ----------------------------- Visualization Functions -----------------------------
def plot_rating_distribution(user_ratings, title='Rating Distribution'):
    fig = px.histogram(user_ratings, x='rating', nbins=10,
                       title=title,
                       labels={'rating': 'Rating', 'count': 'Frequency'},
                       category_orders={'rating': list(range(1, 11))},
                       text_auto=True,
                       template='plotly_white')
    fig.update_xaxes(range=[0.5, 10.5])
    fig.update_layout(bargap=0.1)
    return fig


def plot_genre_wordcloud_from_movies(movies_subset, title='Genre Word Cloud'):
    all_genres = []
    for gl in movies_subset['genres_list']:
        all_genres.extend(gl)
    if not all_genres:
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_genres))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    return fig


def plot_genre_avg_rating(user_ratings, movies_df):
    merged = user_ratings.merge(movies_df[['movie_id', 'genres_list']], on='movie_id')
    genre_ratings = []
    for _, row in merged.iterrows():
        for genre in row['genres_list']:
            genre_ratings.append({'genre': genre, 'rating': row['rating']})
    if not genre_ratings:
        return None
    df_genre = pd.DataFrame(genre_ratings)
    genre_stats = df_genre.groupby('genre').agg(
        avg_rating=('rating', 'mean'),
        count=('rating', 'size')
    ).reset_index().sort_values('avg_rating', ascending=True)

    fig = px.bar(genre_stats, y='genre', x='avg_rating', orientation='h',
                 title='Average Rating by Genre',
                 labels={'avg_rating': 'Average Rating', 'genre': ''},
                 text='avg_rating', hover_data={'count': True},
                 color='avg_rating', color_continuous_scale='Blues',
                 template='plotly_white')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def plot_rating_timeline(ratings_df, freq='D', title='Ratings Over Time'):
    if ratings_df.empty:
        return None
    timeline = ratings_df.sort_values('timestamp').groupby(
        ratings_df['timestamp'].dt.to_period(freq)).size().reset_index(name='count')
    timeline['timestamp'] = timeline['timestamp'].astype(str)
    fig = px.line(timeline, x='timestamp', y='count', title=title,
                  labels={'timestamp': 'Time', 'count': 'Number of Ratings'},
                  template='plotly_white')
    fig.update_layout(height=400)
    return fig


def plot_user_activity_distribution(ratings_df):
    user_counts = ratings_df['user_id'].value_counts().reset_index()
    user_counts.columns = ['user_id', 'rating_count']
    fig = px.histogram(user_counts, x='rating_count', nbins=50,
                       title='User Activity Distribution',
                       labels={'rating_count': 'Number of Ratings per User', 'count': 'Number of Users'},
                       template='plotly_white')
    fig.update_layout(height=400)
    return fig


def plot_year_vs_rating(user_ratings, movies_df):
    merged = user_ratings.merge(movies_df[['movie_id', 'clean_title', 'year', 'genres']], on='movie_id')
    merged = merged[merged['year'] > 0]
    if merged.empty:
        return None
    fig = px.scatter(merged, x='year', y='rating',
                     hover_data={'clean_title': True, 'genres': True},
                     title='Rating vs Release Year',
                     labels={'year': 'Release Year', 'rating': 'Rating'},
                     template='plotly_white',
                     color='rating', color_continuous_scale='Viridis')
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(height=400)
    return fig


# ----------------------------- User Interface: New User (Cold Start + AI only) -----------------------------
def new_user_interface(agent):
    st.header("👤 New User Cold Start")
    tab_names = ["🏠 Cold Start Recommendations", "🤖 AI Assistant"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.subheader("Cold Start Recommendations")
        cold_options = ["Latest Movies", "Most Popular", "Mixed", "Select by Tags"]
        cold_choice = st.radio("Choose recommendation method", cold_options, horizontal=True, key="cold_choice_new")

        if cold_choice == "Select by Tags":
            selected_tags = st.multiselect("Select your favorite movie genres", agent.all_tags, default=[])
            if st.button("Get Recommendations", key="cold_tag_new"):
                if selected_tags:
                    recs, reason = agent.recommend_by_tags(selected_tags, top_k=10)
                    if recs.empty:
                        st.warning("No movies found matching your tags. Try other tags.")
                    else:
                        st.info(f"💡 {reason}")
                        st.dataframe(recs.rename(columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres'}))
                else:
                    st.warning("Please select at least one tag.")
        else:
            type_map = {"Latest Movies": "latest", "Most Popular": "popular", "Mixed": "both"}
            recs, reason = agent.cold_start_recommend(rec_type=type_map[cold_choice], top_k=10)
            st.info(f"💡 {reason}")
            st.dataframe(recs.rename(columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres'}))

    with tabs[1]:
        st.subheader("🤖 AI Assistant Recommendations")
        query = st.text_input("Enter a movie theme or description (e.g., sci-fi movies 1994)", key="new_user_query")
        if st.button("Get AI Recommendations", key="new_user_ai"):
            if query:
                recs, reason = agent.recommend_by_query(query, top_k=10)
                if recs.empty:
                    st.warning("No movies found matching your query. Try other keywords.")
                else:
                    st.info(f"💡 {reason}")
                    st.dataframe(recs.rename(columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres'}))
            else:
                st.warning("Please enter a query.")


# ----------------------------- User Interface: Returning User (Personalized + Franchise + AI + Cold Start) -----------------------------
def returning_user_interface(agent, user_id, ratings):
    user_ratings = ratings[ratings['user_id'] == user_id]
    st.success(f"👋 Welcome back, user {user_id}! You have {len(user_ratings)} ratings.")

    tab_names = ["🧑 Personalized Recommendations", "🎥 Franchise Recommendations", "🤖 AI Assistant", "🏠 Cold Start Recommendations"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.subheader(f"Personalized Recommendations for User {user_id}")
        with st.spinner("Generating user profile..."):
            recs, profile = agent.personalized_recommend(user_id, top_k=10)
        st.write("🧠 User Profile:", profile)
        st.dataframe(recs.rename(columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres'}))

    with tabs[1]:
        st.subheader("Franchise Movie Recommendations")
        user_movies = agent.movies[agent.movies['movie_id'].isin(user_ratings['movie_id'])]
        movie_options = {f"{row['clean_title']} ({row['year']})": row['movie_id']
                         for _, row in user_movies.iterrows()}
        if movie_options:
            selected_movie_title = st.selectbox("Select a movie you've seen", list(movie_options.keys()))
            movie_id = movie_options[selected_movie_title]
            if st.button("Recommend Franchise Movies"):
                recs, reason = agent.franchise_recommend(movie_id)
                st.info(f"💡 {reason}")
                if recs.empty:
                    st.write("No obvious franchise found. Here are similar genre recommendations.")
                else:
                    st.dataframe(recs.rename(columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres'}))
        else:
            st.info("No movie history.")

    with tabs[2]:
        st.subheader("🤖 AI Assistant Recommendations")
        query = st.text_input("Enter a movie theme or description", key="return_user_query")
        if st.button("Get AI Recommendations", key="return_user_ai"):
            if query:
                recs, reason = agent.recommend_by_query(query, top_k=10)
                if recs.empty:
                    st.warning("No movies found matching your query. Try other keywords.")
                else:
                    st.info(f"💡 {reason}")
                    st.dataframe(recs.rename(columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres'}))
            else:
                st.warning("Please enter a query.")

    with tabs[3]:
        st.subheader("Cold Start Recommendations (Auxiliary)")
        cold_options = ["Latest Movies", "Most Popular", "Mixed", "Select by Tags"]
        cold_choice = st.radio("Choose recommendation method", cold_options, horizontal=True, key="cold_choice_old")
        if cold_choice == "Select by Tags":
            selected_tags = st.multiselect("Select your favorite movie genres", agent.all_tags, default=[])
            if st.button("Get Recommendations", key="cold_tag_old"):
                if selected_tags:
                    recs, reason = agent.recommend_by_tags(selected_tags, top_k=10)
                    if recs.empty:
                        st.warning("No movies found matching your tags. Try other tags.")
                    else:
                        st.info(f"💡 {reason}")
                        st.dataframe(recs.rename(columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres'}))
                else:
                    st.warning("Please select at least one tag.")
        else:
            type_map = {"Latest Movies": "latest", "Most Popular": "popular", "Mixed": "both"}
            recs, reason = agent.cold_start_recommend(rec_type=type_map[cold_choice], top_k=10)
            st.info(f"💡 {reason}")
            st.dataframe(recs.rename(columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres'}))


# ----------------------------- Admin Dashboard -----------------------------
def admin_dashboard(ratings, movies, agent):
    st.header("📊 Admin Monitoring System")

    # Overall statistics cards
    total_users = ratings['user_id'].nunique()
    total_ratings = len(ratings)
    avg_rating = ratings['rating'].mean()
    total_movies = movies['movie_id'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Users", total_users)
    col2.metric("Total Ratings", total_ratings)
    col3.metric("Average Rating", f"{avg_rating:.2f}")
    col4.metric("Total Movies", total_movies)

    # First row: rating distribution + user activity distribution
    col_left, col_right = st.columns(2)
    with col_left:
        fig1 = plot_rating_distribution(ratings, title='Overall Rating Distribution')
        st.plotly_chart(fig1, use_container_width=True)
    with col_right:
        fig2 = plot_user_activity_distribution(ratings)
        st.plotly_chart(fig2, use_container_width=True)

    # Second row: daily rating trend + genre word cloud
    col_left2, col_right2 = st.columns(2)
    with col_left2:
        fig3 = plot_rating_timeline(ratings, freq='M', title='Monthly Ratings')
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No timestamp data")
    with col_right2:
        fig4 = plot_genre_wordcloud_from_movies(movies, title='Most Popular Genres (All Movies)')
        if fig4:
            st.pyplot(fig4, use_container_width=True)
        else:
            st.info("No genre data")

    # Text analysis
    st.markdown("---")
    st.subheader("📝 Text Analysis Report")

    # Most active users
    top_active = ratings['user_id'].value_counts().head(5)
    st.write("**Top 5 Most Active Users**:")
    for uid, count in top_active.items():
        st.write(f"- User {uid}: {count} ratings")

    # Highest rated movies (avg, with at least 10 ratings)
    movie_avg = ratings.groupby('movie_id')['rating'].agg(['mean', 'count']).reset_index()
    movie_avg = movie_avg[movie_avg['count'] >= 10].sort_values('mean', ascending=False).head(5)
    movie_avg = movie_avg.merge(movies[['movie_id', 'clean_title', 'year']], on='movie_id')
    st.write("**Top 5 Best Rated Movies (≥10 ratings)**：")
    for _, row in movie_avg.iterrows():
        st.write(f"- {row['clean_title']} ({row['year']}): avg {row['mean']:.2f} ({row['count']} ratings)")

    # Most popular genres
    all_genres = []
    for gl in movies['genres_list']:
        all_genres.extend(gl)
    genre_counts = Counter(all_genres).most_common(5)
    st.write("**Top 5 Most Popular Genres**:")
    for genre, cnt in genre_counts:
        st.write(f"- {genre}: appears in {cnt} movies")

    st.write(
        f"**Overall rating trend**: Average rating {avg_rating:.2f}, standard deviation {ratings['rating'].std():.2f}, median {ratings['rating'].median():.2f}.")

    # ---------- User Detail Monitoring (new) ----------
    st.markdown("---")
    st.subheader("🔍 User Detail Monitoring")

    # Compute rating counts per user for filtering
    user_rating_counts = ratings['user_id'].value_counts()
    max_ratings = int(user_rating_counts.max())
    min_ratings = st.slider("Minimum number of ratings (filter active users)", min_value=1, max_value=max_ratings, value=5, step=1,
                            key="admin_min_ratings")
    filtered_users = user_rating_counts[user_rating_counts >= min_ratings].index.tolist()
    if not filtered_users:
        st.warning("No users satisfy the condition. Showing all users.")
        filtered_users = user_rating_counts.index.tolist()

    selected_user = st.selectbox("Select user ID to view details", sorted(filtered_users), key="admin_user_select")

    if selected_user:
        user_ratings = ratings[ratings['user_id'] == selected_user]
        if user_ratings.empty:
            st.info("This user has no rating history.")
        else:
            st.write(f"**User {selected_user} rating count**: {len(user_ratings)}")

            # Statistics cards
            col_a, col_b, col_c, col_d = st.columns(4)
            avg_u = user_ratings['rating'].mean()
            std_u = user_ratings['rating'].std()
            median_u = user_ratings['rating'].median()
            earliest_date = user_ratings['timestamp'].min().strftime('%Y-%m-%d')
            col_a.metric("Average", f"{avg_u:.2f}")
            col_b.metric("Std Dev", f"{std_u:.2f}" if not pd.isna(std_u) else "N/A")
            col_c.metric("Median", f"{median_u:.2f}")
            col_d.metric("Earliest Rating", earliest_date)

            # Two columns: rating distribution + average rating by genre
            col1, col2 = st.columns(2)
            with col1:
                fig_u1 = plot_rating_distribution(user_ratings, title=f'User {selected_user} Rating Distribution')
                st.plotly_chart(fig_u1, use_container_width=True)
            with col2:
                fig_u2 = plot_genre_avg_rating(user_ratings, movies)
                if fig_u2:
                    st.plotly_chart(fig_u2, use_container_width=True)
                else:
                    st.info("No genre data")

            # Word cloud
            fig_word = plot_genre_wordcloud_from_movies(
                movies[movies['movie_id'].isin(user_ratings['movie_id'])],
                title=f'User {selected_user} Genre Word Cloud'
            )
            if fig_word:
                st.pyplot(fig_word, use_container_width=True)
            else:
                st.info("Insufficient genre data for word cloud")

            # Third row: timeline + year vs rating
            col3, col4 = st.columns(2)
            with col3:
                fig_u3 = plot_rating_timeline(user_ratings, freq='M', title=f'User {selected_user} Ratings Over Time')
                if fig_u3:
                    st.plotly_chart(fig_u3, use_container_width=True)
                else:
                    st.write("Insufficient data for timeline.")
            with col4:
                fig_u4 = plot_year_vs_rating(user_ratings, movies)
                if fig_u4:
                    st.plotly_chart(fig_u4, use_container_width=True)
                else:
                    st.write("No valid year data")

            # History table
            with st.expander("View User Rating History"):
                history = user_ratings.merge(movies[['movie_id', 'clean_title', 'year', 'genres']], on='movie_id')
                history = history[['timestamp', 'clean_title', 'year', 'genres', 'rating']].sort_values('timestamp',
                                                                                                        ascending=False)
                st.dataframe(history.rename(
                    columns={'clean_title': 'Title', 'year': 'Year', 'genres': 'Genres', 'rating': 'Rating',
                             'timestamp': 'Time'}))


# ----------------------------- Main Program -----------------------------
def main():
    st.set_page_config(page_title="LLM Movie Agent", layout="wide")
    st.title("🎬 LLM Agent Movie Recommendation System")

    # Load data
    with st.spinner("Loading data..."):
        ratings = load_ratings()
        movies = load_movies()
        embedder = load_embedder()
        faiss_index, _, _ = build_faiss_index(movies, embedder)

    # Initialize Agent
    agent = MovieAgent(movies, ratings, embedder, faiss_index)

    # Sidebar role selection
    st.sidebar.title("Role Selection")
    role = st.sidebar.radio("Select your role", ["Regular User", "Admin"], index=0)

    if role == "Regular User":
        st.sidebar.header("User Login")
        user_type = st.sidebar.radio("User Type", ["New User", "Returning User"], index=0)

        if user_type == "New User":
            # New user directly enters cold start interface
            new_user_interface(agent)
        else:
            # Returning user needs to input ID
            user_id_input = st.sidebar.text_input("Enter user ID (numeric)", value="")
            login_btn = st.sidebar.button("Login")

            if login_btn and user_id_input:
                try:
                    user_id = int(user_id_input)
                    st.session_state['user_id'] = user_id
                    st.session_state['logged_in'] = True
                except:
                    st.sidebar.error("Please enter a valid numeric ID")
            elif login_btn and not user_id_input:
                st.sidebar.warning("Please enter user ID")

            if 'logged_in' in st.session_state and st.session_state['logged_in']:
                user_id = st.session_state['user_id']
                returning_user_interface(agent, user_id, ratings)
            else:
                st.info("Please enter user ID in the sidebar and log in")
    else:
        # Admin interface
        admin_dashboard(ratings, movies, agent)


if __name__ == "__main__":
    main()