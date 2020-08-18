from typing import Union

from pyspark import RDD, Row
from pyspark.sql import DataFrame

from sourced.ml.transformers.transformer import Transformer
from sourced.ml.transformers.uast2bag_features import Uast2BagFeatures

from utils.extended_document_constants import ExtendedDocumentConstants


class UastRow2ExtendedDocument(Transformer):
    """
    Converts UAST rows into document rows with author and repo entries.

    This class is based on sourced.ml.transformers.uast2bag_features.UastRow2Document
    and returns the same doc and uast entries. The difference between
    a doc object and an extended doc object is that the extended doc
    includes separate author and repository id entries to be easily used
    later during training.
    """

    REPO_PATH_SEP = "//"
    PATH_BLOB_SEP = "@"

    def __call__(self, rows: Union[RDD, DataFrame]):
        if isinstance(rows, DataFrame):
            rows = rows.rdd

        return rows.map(self.documentize)

    def documentize(self, r: Row) -> Row:
        ec = ExtendedDocumentConstants.Columns

        doc = r[ec.RepositoryId]
        if r[ec.Path]:
            doc += self.REPO_PATH_SEP + r[ec.Path]
        if r[ec.BlobId]:
            doc += self.PATH_BLOB_SEP + r[ec.BlobId]

        bfc = Uast2BagFeatures.Columns
        return Row(**{bfc.document: doc, ec.RepositoryId: r[ec.RepositoryId],
                      ec.Author: r[ec.Author], ec.Uast: r[ec.Uast]})
