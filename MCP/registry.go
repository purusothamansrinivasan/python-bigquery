package main

import (
	"github.com/bigquery-api/mcp-server/config"
	"github.com/bigquery-api/mcp-server/models"
	tools_tables "github.com/bigquery-api/mcp-server/tools/tables"
	tools_datasets "github.com/bigquery-api/mcp-server/tools/datasets"
	tools_jobs "github.com/bigquery-api/mcp-server/tools/jobs"
	tools_tabledata "github.com/bigquery-api/mcp-server/tools/tabledata"
	tools_routines "github.com/bigquery-api/mcp-server/tools/routines"
	tools_projects "github.com/bigquery-api/mcp-server/tools/projects"
	tools_models "github.com/bigquery-api/mcp-server/tools/models"
	tools_rowaccesspolicies "github.com/bigquery-api/mcp-server/tools/rowaccesspolicies"
)

func GetAll(cfg *config.APIConfig) []models.Tool {
	return []models.Tool{
		tools_tables.CreateBigquery_tables_testiampermissionsTool(cfg),
		tools_datasets.CreateBigquery_datasets_undeleteTool(cfg),
		tools_jobs.CreateBigquery_jobs_cancelTool(cfg),
		tools_datasets.CreateBigquery_datasets_listTool(cfg),
		tools_datasets.CreateBigquery_datasets_insertTool(cfg),
		tools_tabledata.CreateBigquery_tabledata_insertallTool(cfg),
		tools_datasets.CreateBigquery_datasets_deleteTool(cfg),
		tools_datasets.CreateBigquery_datasets_getTool(cfg),
		tools_datasets.CreateBigquery_datasets_patchTool(cfg),
		tools_datasets.CreateBigquery_datasets_updateTool(cfg),
		tools_routines.CreateBigquery_routines_listTool(cfg),
		tools_routines.CreateBigquery_routines_insertTool(cfg),
		tools_jobs.CreateBigquery_jobs_listTool(cfg),
		tools_tables.CreateBigquery_tables_getiampolicyTool(cfg),
		tools_projects.CreateBigquery_projects_listTool(cfg),
		tools_routines.CreateBigquery_routines_deleteTool(cfg),
		tools_routines.CreateBigquery_routines_getTool(cfg),
		tools_routines.CreateBigquery_routines_updateTool(cfg),
		tools_tabledata.CreateBigquery_tabledata_listTool(cfg),
		tools_jobs.CreateBigquery_jobs_getTool(cfg),
		tools_jobs.CreateBigquery_jobs_deleteTool(cfg),
		tools_jobs.CreateBigquery_jobs_queryTool(cfg),
		tools_models.CreateBigquery_models_listTool(cfg),
		tools_models.CreateBigquery_models_deleteTool(cfg),
		tools_models.CreateBigquery_models_getTool(cfg),
		tools_models.CreateBigquery_models_patchTool(cfg),
		tools_tables.CreateBigquery_tables_deleteTool(cfg),
		tools_tables.CreateBigquery_tables_getTool(cfg),
		tools_tables.CreateBigquery_tables_patchTool(cfg),
		tools_tables.CreateBigquery_tables_updateTool(cfg),
		tools_rowaccesspolicies.CreateBigquery_rowaccesspolicies_listTool(cfg),
		tools_tables.CreateBigquery_tables_listTool(cfg),
		tools_tables.CreateBigquery_tables_insertTool(cfg),
		tools_tables.CreateBigquery_tables_setiampolicyTool(cfg),
		tools_jobs.CreateBigquery_jobs_getqueryresultsTool(cfg),
		tools_projects.CreateBigquery_projects_getserviceaccountTool(cfg),
	}
}
