from flask import Flask, request, Response
import redis
from minio import Minio
from minio.error import S3Error
import base64
import json
import hashlib
import io
import os

app = Flask(__name__)

REDIS_HOST = os.getenv('REDIS_HOST') if os.getenv('REDIS_HOST')!=None else "localhost"
REDIS_PORT = os.getenv('REDIS_PORT') if os.getenv('REDIS_PORT')!=None else "6379"
MINIO_HOST = os.getenv('MINIO_HOST') if os.getenv('MINIO_HOST')!=None else "0.0.0.0"
MINIO_PORT = os.getenv('MINIO_PORT') if os.getenv('MINIO_PORT')!=None else "9000"
MINIO_USER = os.getenv('MINIO_USER') if os.getenv('MINIO_USER')!=None else "rootuser"
MINIO_PASSWD = os.getenv('MINIO_PASSWD') if os.getenv('MINIO_PASSWD')!=None else "rootpass123"
GET_HOSTS_FROM = os.getenv('GET_HOSTS_FROM') if os.getenv('GET_HOSTS_FROM')!=None else "dns"

redis_conn = redis.from_url(f'redis://{REDIS_HOST}:{REDIS_PORT}')
minio_conn = Minio(
    endpoint=f"{MINIO_HOST}:{MINIO_PORT}",
    access_key=f"{MINIO_USER}",
    secret_key=f"{MINIO_PASSWD}",
    secure=False
)

@app.route("/")
def index():
    return """
    <h1> Music Separation Server</h1>
    <p> Use a valid endpoint </p>
    """

@app.route('/apiv1/separate', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        r = request.get_json()

        # decode data to binary object
        mp3 = base64.b64decode(r['mp3'])
        callback_url = r['callback']['url']

        # get hash value of object
        hash = hashlib.sha256(mp3).hexdigest()

        # upload object to minio bucket
        if not minio_conn.bucket_exists(bucket_name="songbank"):
            minio_conn.make_bucket(bucket_name="songbank")
        else:
            print("Bucket 'songbank' already exists!")

        minio_conn.put_object(
                bucket_name="songbank",
                object_name=f"{hash}/{hash}.mp3",
                data=io.BytesIO(mp3),
                length=len(mp3)
            )
        print(f"{hash} successfully uploaded as object {hash}.mp3 to bucket 'songbank'!")

        # http://peter-hoffmann.com/2012/python-simple-queue-redis-queue.html
        # add object to list of songs left to process
        redis_conn.rpush('song_queue', hash)
        # add callback key value pair
        redis_conn.set(f'callback_{hash}', callback_url)

        return Response(
            response=json.dumps({
                'hash': hash,
                'reason': "Song enqueued for separation"
            }),
            status=200,
            headers={'Content-type': 'application/json'}
        )
    else:
        return 'Upload files to this endpoint!'

@app.route('/apiv1/queue', methods=['GET'])
def show_queue():
    """
    Return list of hashes that are queued and pending processing.
    """

    queue = []
    # fetch list of queued songs
    queue += redis_conn.lrange('song_queue', 0, -1)
    # fetch list of processed songs
    queue += list(redis_conn.smembers('song_list'))

    # convert binary hashes to string
    queue = [s.decode() for s in queue]

    return Response(
        response=json.dumps({'queue': queue}),
        status=200,
        headers={'Content-type': 'application/json'}
    )

@app.route('/apiv1/track/<string:track>', methods=['GET'])
def fetch_track(track):
    """
    Retrieve the track (any of base.mp3, vocals.mp3, drums.mp3, other.mp3) as a binary download.
    For example, you should be able to redirect the output of a curl command to a file and play that file as an MP3 file.
    """
    # get song hash and isolation type
    hash, track = track.split('_')

    try:
        # fetch track from minio bucket
        object = minio_conn.get_object(bucket_name='songbank', object_name=f'{hash}/separated/{track}.wav')
        data = object.data
        object.close()
        object.release_conn()
    except S3Error:
        # if track is not found
        return Response(
            response=json.dumps({
                'hash': hash,
                'reason': "Song not found"
            }),
            status=404,
            headers={'Content-type': 'application/json'}
        )

    return Response(
        response=io.BytesIO(data),
        status=200,
        headers={'Content-type': 'audio/wav'}
    )

@app.route('/apiv1/remove/<string:hash>', methods=['GET'])
def remove_track(hash):
    """
    Remove specified track.
    """

    # remove song from pending and processed queues
    queue_res = redis_conn.lrem('song_queue', 1, hash) # res is 1 when something is deleted and 0 otherwise
    list_res = redis_conn.srem('song_list', 1, hash) # res is 1 when something is deleted and 0 otherwise

    # remove callback urls
    callback_res = redis_conn.delete(f'callback_{hash}')

    # remove song and separated tracks from bucket
    objects_to_delete = minio_conn.list_objects(bucket_name='songbank', prefix=hash, recursive=True)
    for obj in objects_to_delete:
        minio_conn.remove_object(bucket_name='songbank', object_name=obj.object_name)
    
    # if no songs found
    if queue_res==0 and list_res==0:
        return Response(
            response=json.dumps({
                'hash': hash,
                'reason': "Song not found"
            }),
            status=404,
            headers={'Content-type': 'application/json'}
        )

    return Response(
        response=json.dumps({
            'hash': hash,
            'reason': "Song deleted"
        }),
        status=200,
        headers={'Content-type': 'application/json'}
    )

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)