from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import random

app = FastAPI(title="Reddit API")

class Post(BaseModel):
    id: int
    title: str
    content: str
    upvotes: int = 0
    
posts = [
    Post(id=1, title="First post", content="Hello Reddit!", upvotes=5),
    Post(id=2, title="Second post", content="Another post", upvotes=3),
    Post(id=3, title="Third post", content="Final test post", upvotes=8)
]

@app.get("/")
def read_root():
    return {"message": "Welcome to Reddit API"}

@app.get("/posts", response_model=List[Post])
def get_posts():
    return posts

@app.get("/posts/{post_id}", response_model=Post)
def get_post(post_id: int):
    for post in posts:
        if post.id == post_id:
            return post
    raise HTTPException(status_code=404, detail="Post not found")

@app.post("/posts", response_model=Post)
def create_post(post: Post):
    posts.append(post)
    return post

@app.put("/posts/{post_id}/upvote")
def upvote_post(post_id: int):
    for post in posts:
        if post.id == post_id:
            post.upvotes += 1
            return {"message": "Post upvoted successfully"}
    raise HTTPException(status_code=404, detail="Post not found")