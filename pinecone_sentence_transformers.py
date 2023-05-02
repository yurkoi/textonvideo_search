from sentence_transformers import SentenceTransformer
import pandas as pd
import pinecone


API_KEY = '1d3e38ba-e7a8-4e32-aca7-5e75da810365'

pinecone.init(api_key=API_KEY, environment='us-east1-gcp')
model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
print(f"Models encoding size: {model.get_sentence_embedding_dimension()}")
index = pinecone.Index('video-search1')


def upsert_data(index):
    data = pd.read_csv('df_tofind.csv').dropna()
    if 'video-search1' not in pinecone.list_indexes():
        pinecone.create_index(
            name='video-search1', dimension=model.get_sentence_embedding_dimension(), metric='cosine')

    upserts = [(str(v[0]), model.encode(v[3]).tolist(),
                {'text': v[3], 'start': v[1], 'end': v[2]}) for v in data.values]
    # print(upserts)
    index.upsert(vectors=upserts)


def parse_ans(answer):
    return {
        # 'result': answer,
        'time_start': answer['matches'][0]['metadata']['start'].time(),
        'time_end': answer['matches'][0]['metadata']['end'].time(),
        'text': answer['matches'][0]['metadata']['text']
    }


def get_query(query, ind=index):
    xq = model.encode([query]).tolist()
    result = ind.query(xq, top_k=2, include_metadata=True)
    return parse_ans(result)


if __name__ == "__main__":
    # upsert_data(index)
    result = get_query('masturbation spirit', index)
    print(parse_ans(result))

