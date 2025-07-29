import asyncio
import json
import pytest
import websockets

from trading_bot.liquidity_ws import LiquidityStream


@pytest.mark.asyncio
async def test_liquidity_stream_receives_messages():
    """Comprueba que LiquidityStream.listen() procesa mensajes JSON."""

    async def echo_server(websocket):
        msg = json.dumps({
            "symbol": "BTC_USDT",
            "bids": [[100, 1]],
            "asks": [[101, 1]]
        })
        await websocket.send(msg)

    server = await websockets.serve(echo_server, "localhost", 12345)

    stream = LiquidityStream()
    consumer_task = asyncio.create_task(
        stream.listen(["BTC_USDT"], url="ws://localhost:12345")
    )

    await asyncio.sleep(0.5)
    await stream.stop()
    consumer_task.cancel()
    server.close()
    await server.wait_closed()

    assert "BTC_USDT" in stream._orderbook
    book = stream._orderbook["BTC_USDT"]
    assert book["bids"] == [[100.0, 1.0]]
    assert book["asks"] == [[101.0, 1.0]]
