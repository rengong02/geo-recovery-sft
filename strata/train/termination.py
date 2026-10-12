"""Shared validation of the public v2 termination payload."""


def valid_v2_termination(payload, family: str) -> bool:
    if not isinstance(payload, dict):
        return False
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 2:
        return False
    status = payload.get("status")
    if status == "error":
        return all(isinstance(payload.get(key), str) and payload[key].strip()
                   for key in ("error_code", "error_message"))
    if status != "done":
        return False
    if family.upper() == "F2":
        keys = payload.get("join_keys")
        return isinstance(keys, list) and len(keys) == 2 and all(
            isinstance(key, str) and key.strip() for key in keys)
    selected = payload.get("selected_column")
    return isinstance(selected, str) and bool(selected.strip())
