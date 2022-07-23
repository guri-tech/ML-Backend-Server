# from fastapi import FastAPI
#
# app = FastAPI()
#
# @app.get('/items/')
# def read_items(q: str | None = None):
#     results = {'items':[{'item_id':'Foo'},{'item_id':'Bar'}]}
#     if q:
#         results.update({'q':q})
#     return results

# from fastapi import FastAPI,Query
# from typing import Optional
# app = FastAPI()
#
#
# @app.get("/items/")
# async def read_items(q: str or None = Query(default= None, max_length= 50)):
#     results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
#     if q:
#         results.update({"q": q})
#     return results

from typing import Optional

import numpy as np
from fastapi import FastAPI

# from fastapi import FastAPI
#
# app = FastAPI()
# fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]
#
# @app.get('/items/')
# def read_item(skip: int = 0, limit: int = 10):
#     return fake_items_db[skip: skip + limit]

# from typing import Optional
# from fastapi import FastAPI
#
# app = FastAPI()
#
# @app.get('/items/{item_id}')
# def read_item(item_id:str,q:Optional[str] =None):
#     if q:
#         return{'item_id':item_id, 'q':q}
#     return {'item_id':item_id}

# from typing import Optional
# from fastapi import FastAPI
#
# app = FastAPI()
#
# @app.get('/items/{item_id}')
# def read_item(item_id: str, q: Optional[str] = None,short : bool = False):
#     item =  {'item_id':item_id}
#     if q:
#         item.update({'q':q})
#         if not short:
#             item.update(
#                 {'description': 'this is an amazing item that has a long '
#                                 'description'}
#             )
#         return item

# from typing import Optional
# from fastapi import FastAPI
#
# app = FastAPI()
#
# @app.get('/users/{user_id}/items/{item_id}')
# def read_user_item(
#         user_id : int, item_id: str, q: Optional[str] = None, short: bool = False
# ):
#     item = {'item_id':item_id,'owner_id':user_id}
#     if q:
#         item.update({'q':q})
#     if not short:
#         item.updata(
#             {'description':'this is an amazing item that has a long description'}
#         )
#     return item

# from fastapi import FastAPI
#
# app = FastAPI()
#
# @app.get('/items/{item_id}')
# def read_user_item(item_id: str,needy: str):
#     item = {'item_id':item_id,'needy':needy}
#     return item

from fastapi import FastAPI
import predict
app = FastAPI()
@app.get('/')
def root():
    return '기계학습 Iris 품종 분류 예측하기'



app.include_router(predict.router)




































































