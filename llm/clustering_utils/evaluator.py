import logging
import numpy as np
from transformers import AutoTokenizer
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def clustering_score(y_true, y_pred):
    return {'ACC': clustering_accuracy_score(y_true, y_pred) * 100,
            'ARI': adjusted_rand_score(y_true, y_pred) * 100,
            'NMI': normalized_mutual_info_score(y_true, y_pred) * 100}


DEFINITIONS = {
    'hkunlp/instructor-xl': {
        "dolphin": "Represent the social connections for clustering: ",
        "football": "Represent the game schedule for clustering: ",
        "karate": "Represent the friendship ties for clustering: ",
        "mexican": "Represent the social interactions for clustering: ",
        "polbooks": "Represent the book co-purchases for clustering: ",
        "railway": "Represent the station connections for clustering: ",
        "strike": "Represent the collaborative actions for clustering: ",
        "TC": "Represent the social interactions for clustering: ",  # node connections
    },
    'hkunlp/instructor-large': {
        "dolphin": "Represent the social connections for clustering: ",
        "football": "Represent the game schedule for clustering: ",
        "karate": "Represent the friendship ties for clustering: ",
        "mexican": "Represent the social interactions for clustering: ",
        "polbooks": "Represent the book co-purchases for clustering: ",
        "railway": "Represent the station connections for clustering: ",
        "strike": "Represent the collaborative actions for clustering: ",
        "TC": "Represent the social interactions for clustering: ",  # node connections
    },
    'hkunlp/instructor-base': {
        "dolphin": "Represent the social connections for clustering: ",
        "football": "Represent the game schedule for clustering: ",
        "karate": "Represent the friendship ties for clustering: ",
        "mexican": "Represent the social interactions for clustering: ",
        "polbooks": "Represent the book co-purchases for clustering: ",
        "railway": "Represent the station connections for clustering: ",
        "strike": "Represent the collaborative actions for clustering: ",
        "TC": "Represent the social interactions for clustering: ",  # node connections
    },
}


class ClusteringEvaluator(object):
    def __init__(self, sentences, labels, clustering_batch_size=500, limit=None, **kwargs):
        # super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.args = kwargs['args']
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        new_sentences = []
        if self.args.prompt:
            print('with prompt')
            for s in self.sentences:
                if len(self.tokenizer(DEFINITIONS[self.args.prompt][self.args.task_name] + s)['input_ids']) <= 256:
                    new_sentences.append([DEFINITIONS[self.args.prompt][self.args.task_name], s, 0])
                else:
                    new_sentences.append(['', s, 0])
        else:
            new_sentences = self.sentences
        corpus_embeddings = np.asarray(model.encode(new_sentences))
        # mean_emb = np.mean(corpus_embeddings,axis=0)
        # corpus_embeddings -= mean_emb

        if self.labels is not None:
            label_ids, n_clusters = self._convert_label_to_ids(self.labels)

            all_measures = {'ACC': [], 'NMI': [], 'ARI': []}
            # for seed in [100, 13, 21, 36, 42]:
            for seed in [13]:
                if self.args.scale == "sentence":  # our case
                    logger.info(f"Fitting K-Means model (seed: {seed})...")
                    preds = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(corpus_embeddings)
                if self.args.scale == "small":
                    logger.info(f"Fitting K-Means model (seed: {seed})...")
                    preds = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(corpus_embeddings)
                elif self.args.scale == "large":
                    logger.info(f"Fitting MiniBatch K-Means model (seed: {seed})...")
                    preds = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed).fit_predict(corpus_embeddings)
                preds = np.asarray(preds)
                measures = clustering_score(label_ids, preds)
                for k in measures:
                    all_measures[k].append(measures[k])

            for k in ['ACC', 'NMI', 'ARI']:
                # print(k)
                mean = np.mean(all_measures[k])
                # print("Mean: ", round(mean, 2))
                std = np.std(all_measures[k])
                # print("Std: ", round(std, 2))

                all_measures[f'{k}_mean'] = mean
                all_measures[f'{k}_std'] = std

        else:
            all_measures = {}

        return all_measures, corpus_embeddings

    def eval_only(self, corpus_embeddings):
        if self.labels is not None:
            label_ids, n_clusters = self._convert_label_to_ids(self.labels)

            all_measures = {'ACC': [], 'NMI': [], 'ARI': []}
            # for seed in [100, 13, 21, 36, 42]:
            for seed in [13]:
                if self.args.scale == "sentence":  # our case
                    logger.info(f"Fitting K-Means model (seed: {seed})...")
                    preds = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(corpus_embeddings)
                elif self.args.scale == "small":
                    logger.info(f"Fitting K-Means model (seed: {seed})...")
                    preds = KMeans(n_clusters=n_clusters, random_state=seed).fit_predict(corpus_embeddings)
                elif self.args.scale == "large":
                    logger.info(f"Fitting MiniBatch K-Means model (seed: {seed})...")
                    preds = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed).fit_predict(corpus_embeddings)
                preds = np.asarray(preds)
                measures = clustering_score(label_ids, preds)
                for k in measures:
                    all_measures[k].append(measures[k])

            for k in ['ACC', 'NMI', 'ARI']:
                # print(k)
                mean = np.mean(all_measures[k])
                # print("Mean: ", round(mean, 2))
                std = np.std(all_measures[k])
                # print("Std: ", round(std, 2))

                all_measures[f'{k}_mean'] = mean
                all_measures[f'{k}_std'] = std
        else:
            all_measures = {}

        return all_measures

    def _convert_label_to_ids(self, labels):
        unique_labels = list(set(labels))
        n_clusters = len(unique_labels)
        label_map = {l: i for i, l in enumerate(unique_labels)}
        label_ids = [label_map[l] for l in labels]
        return np.asarray(label_ids), n_clusters
