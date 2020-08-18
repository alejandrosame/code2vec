from pyspark.sql import DataFrame

from sourced.engine import Engine
from sourced.ml.transformers.transformer import Transformer

from utils.blob_constants import BlobConstants
from utils.commit_author_constants import CommitAuthorConstants


class Deanonymizer(Transformer):
    """
    Adds author id to blob row.

    The author id is added by joining the input blobs dataframe
    with the commits dataframe by commit hash. The commits dataframe
    needs to be recovered from the root engine object specified
    in the constructor.
    """

    def __init__(self, engine: Engine, **kwargs):
        super().__init__(**kwargs)
        self.engine = engine

    def __call__(self, rows: DataFrame):
        return self.deanonymize(rows)

    def deanonymize(self, d: DataFrame) -> DataFrame:
        hash_authors = self.engine.repositories.references \
                                  .head_ref.commits \
                                  .select("hash", "author_email")

        bc = BlobConstants.Columns
        cac = CommitAuthorConstants.Columns

        deanonymized = \
            d.join(hash_authors,
                   (d[bc.CommitHash] == hash_authors[cac.CommitHash]))\
             .drop(cac.CommitHash)

        return deanonymized
