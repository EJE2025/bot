"""Entry point for the analytics service exposing a GraphQL API."""
from __future__ import annotations

from flask import Flask
from flask_graphql import GraphQLView

from .api.graphql import schema
from .database import Base, engine


def create_app() -> Flask:
    app = Flask(__name__)
    Base.metadata.create_all(bind=engine)

    app.add_url_rule(
        "/graphql",
        view_func=GraphQLView.as_view("graphql", schema=schema, graphiql=True),
    )
    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
