from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
import environment

redisai_client = environment.redis_r()


def redis_setting_call(train_model, redis_vari_num):

    redisai_client.set("new_model_name", str(train_model.__hash__()))

    initial_inputs = [("float_input", FloatTensorType([None, redis_vari_num]))]
    onnx_model = convert_sklearn(
        train_model, initial_types=initial_inputs, target_opset=12
    )

    convert_model_name = redisai_client.get("new_model_name")
    redisai_client.modelstore(
        key=convert_model_name,
        backend="onnx",
        device="cpu",
        data=onnx_model.SerializeToString(),
    )
