PUBLIC_VIEWS = frozenset(
    {
        "hydrology_monitoring_devices",
        "hydrology_single_factor_alarms",
        "hydrology_multifactor_warnings",
    }
)

PUBLIC_CUBES = frozenset(
    {
        "base_device_info",
        "base_device_x_value",
        "base_label",
        "base_label_sensor",
        "base_multifactor_sensor",
        "base_warn_state_info",
        "base_water_warn_sensor_set",
    }
)

PUBLIC_MODELS = PUBLIC_VIEWS | PUBLIC_CUBES

CUBE_JOIN_EDGES = {
    "base_device_info": frozenset(),
    "base_device_x_value": frozenset(
        {"base_device_info", "base_label_sensor"}
    ),
    "base_label": frozenset(),
    "base_label_sensor": frozenset({"base_label"}),
    "base_multifactor_sensor": frozenset(
        {"base_water_warn_sensor_set", "base_device_x_value"}
    ),
    "base_warn_state_info": frozenset(
        {"base_device_x_value", "base_water_warn_sensor_set"}
    ),
    "base_water_warn_sensor_set": frozenset(),
}

PRIVATE_MEMBERS = frozenset(
    {
        "base_device_info.svg_position",
        "base_device_info.port",
        "base_device_x_value.analysis_id",
        "base_device_x_value.register_address",
        "base_device_x_value.register_count",
        "base_device_x_value.start_index",
        "base_device_x_value.length",
        "base_label.created_by_id",
        "base_label.updated_by_id",
        "base_multifactor_sensor.historical_data_path",
        "base_multifactor_sensor.created_by_id",
        "base_multifactor_sensor.updated_by_id",
        "base_water_warn_sensor_set.prediction_data_path",
        "base_water_warn_sensor_set.created_by_id",
        "base_water_warn_sensor_set.updated_by_id",
        "base_water_warn_sensor_set.coefficient_data_path",
        "base_water_warn_sensor_set.prediction_function_id",
        "base_water_warn_sensor_set.prediction_function",
        "base_water_warn_sensor_set.prediction_parameters",
        "base_water_warn_sensor_set.aggregation_script",
        "base_water_warn_sensor_set.aggregation_script_id",
        "base_water_warn_sensor_set.judgment_script_id",
        "base_water_warn_sensor_set.judgment_script",
        "base_water_warn_sensor_set.value_conversion_script",
        "base_water_warn_sensor_set.value_conversion_script_id",
    }
)

STRING_MEMBERS = frozenset(
    {
        "hydrology_monitoring_devices.sensor_id",
        "hydrology_monitoring_devices.device_id",
        "hydrology_monitoring_devices.label_id",
        "hydrology_single_factor_alarms.alarm_event_id",
        "hydrology_single_factor_alarms.sensor_id",
        "hydrology_single_factor_alarms.device_id",
        "hydrology_multifactor_warnings.warning_event_id",
        "hydrology_multifactor_warnings.configuration_id",
        "base_device_info.id",
        "base_device_x_value.id",
        "base_device_x_value.device_id",
        "base_label.id",
        "base_label.parent_id",
        "base_label_sensor.relation_key",
        "base_label_sensor.sensor_id",
        "base_label_sensor.label_id",
        "base_multifactor_sensor.id",
        "base_multifactor_sensor.parent_id",
        "base_multifactor_sensor.sensor_id",
        "base_warn_state_info.id",
        "base_warn_state_info.source_id",
        "base_water_warn_sensor_set.id",
    }
)
