"""Worker enqueue client.

Thin proxy over a Dramatiq broker. The app side uses this to dispatch tasks
to the worker without importing any worker-side code.

Usage:
    from .worker import later
    later.process_document(task_id)
"""
from dramatiq.brokers.redis import RedisBroker
from dramatiq import Message

from . import config


class Later:
    """
    Dispatch tasks to Dramatiq actors by attribute access.

    Any attribute access returns a callable that enqueues to that actor.
    The caller sees no Message concept — it just calls a function.

    Example:
        later = Later(host="localhost", port=6380)
        later.process_document("task-abc123")
        later.add(10, 20)
    """

    def __init__(self, host: str, port: int, default_queue: str = "default"):
        self._broker = RedisBroker(host=host, port=port)
        self._default_queue = default_queue

    def __getattr__(self, actor_name: str):
        queue = self._default_queue

        def _enqueue(*args, **kwargs):
            msg = Message(
                queue_name=queue,
                actor_name=actor_name,
                args=list(args),
                kwargs=kwargs,
                options={},
            )
            self._broker.enqueue(msg)
            return msg.message_id

        _enqueue.__name__ = actor_name
        return _enqueue


later = Later(host=config.REDIS_HOST, port=config.REDIS_PORT)
