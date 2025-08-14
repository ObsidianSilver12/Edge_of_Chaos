# wbs matrix only applies to nodes as fragments will not have a wbs initially, this will 
# be temporarily assigned during subconscious processing and only permanently assigned 
# during conscious processsing

wbs_status = {
    "wbs_status_ID": "UUID",
    "wbs_status_title": "str",  # e.g. active, inactive, archived
    "wbs_status_description": "str"  # description of the status
}

# joined table not a query as it has to be maintained
wbs_matrix_active = {
    "wbs_matrix_active_ID": "UUID",
    "wbs_active": [], # array of all active wbs from wbs level 1,2 and 3
    "wbs_active_count": 0,  # count of active wbs, used for quick access
    "fk_wbs_status_id": ""  # status of the wbs, can
}

# joined table not a query as it has to be maintained
wbs_matrix_inactive = {
    "wbs_matrix_inactive_ID": "UUID",
    "wbs_all_inactive": "", # array of all inactive wbs from wbs level 1,2 and 3
    "wbs_inactive_count": 0,  # count of inactive wbs, used for quick access
    "fk_wbs_status_id": ""  # status of the wbs, can be active, inactive, archived
}

wbs_matrix_archived = {
    "wbs_matrix_archived_ID": "UUID",
    "wbs_all_archived": [],  # array of all archived wbs from wbs level 1,2 and 3
    "wbs_archived_count": 0,  # count of archived wbs, used for quick access
    "fk_wbs_status_id": ""  # status of the wbs, can be active, inactive, archived
}

# joined table not a query as it has to be maintained
hierarchy_structure = {
    "hierarchy_structure_ID": "",
    "fk_main_categories_ID": "",
    "fk_wbs_lvl_1_ID": "",
    "fk_domain_sub_categories_ID": "",
    "fk_wbs_lvl_2_ID": "",
    "fk_wbs_lvl_3_ID": ""
}

wbs_structure ={
    "wbs_structure_ID": "UUID",
    "fk_wbs_lvl_1_ID": "",
    "fk_wbs_lvl_2_ID": "",
    "fk_wbs_lvl_3_ID": ""
}

# Table for dropdowns populated with some basic categories but expanded as more data is collected
main_categories = {
    "main_categories_ID":"UUID",
    "main_categories_title":"str",
    "main_categories_description":"str", # description of the main category
    "fk_wbs_lvl_1_ID": ""  # the associated lvl 1 wbs, used for grouping
}


wbs_level_1 = {
    "wbs_lvl_1_ID": "UUID",
    "wbs_lvl_1_title":"str",# domain category
    "wbs_lvl_1_status":"", # list inactive or active number or bool which is most memory efficient
    "fk_domain_sub_categories_ID": "" # domain sub-category populated after from domain sub category dropdowns
}

domain_sub_categories = {
    "domain_sub_categories_ID": "UUID",
    "domain_sub_categories_title": "str", # sub-category title
    "fk_wbs_lvl_1_ID": "" # main domain associated to sub domain for grouping labelled after the fact
}

wbs_level_2 = {
    "wbs_lvl_2_ID": "UUID",
    "wbs_lvl_2_title":"str",# concept category
    "wbs_lvl_2_status":"", # list inactive or active number or bool which is most memory efficient
    "fk_wbs_lvl_1_ID": "" # the associated lvl 1 wbs
}

wbs_level_3 = {
    "wbs_lvl_3_ID": "UUID",
    "wbs_lvl_3_title":"str", # related concepts category
    "wbs_lvl_3_status":"", # list inactive or active number or bool which is most memory efficient
    "fk_wbs_lvl_2_ID": "" # the associated lvl 2 wbs
}




