emotional = {
    "emotional_id": "UUID",
    "fk_fragment_id": "UUID",
    "system_state": "",
    "cpu_usage": ,
    "memory_usage":"" ,
    "temperature_sensors":"" ,
    "power_consumption":"" ,
    "inferred_state":"" , #calculation based on physical processing state - ranges infer stress states and associates value to an emotion
    "measurement_time_start":"" ,
    "measurement_time_end":"" ,
    "measurement_duration":"" ,
    "field_measure": "", # measure local field frequency compare to stored frequency value
    "field_state": "" # based on field frequency assign a range value i.e. stable, discordant, resonant
    "health_trigger": "bool" #trigger a health event if stressed or field is unstable
}