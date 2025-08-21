package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/bigquery-api/mcp-server/config"
	"github.com/bigquery-api/mcp-server/models"
	"github.com/mark3labs/mcp-go/mcp"
)

func Bigquery_tables_getHandler(cfg *config.APIConfig) func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	return func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		args, ok := request.Params.Arguments.(map[string]any)
		if !ok {
			return mcp.NewToolResultError("Invalid arguments object"), nil
		}
		projectIdVal, ok := args["projectId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: projectId"), nil
		}
		projectId, ok := projectIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: projectId"), nil
		}
		datasetIdVal, ok := args["datasetId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: datasetId"), nil
		}
		datasetId, ok := datasetIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: datasetId"), nil
		}
		tableIdVal, ok := args["tableId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: tableId"), nil
		}
		tableId, ok := tableIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: tableId"), nil
		}
		queryParams := make([]string, 0)
		if val, ok := args["selectedFields"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("selectedFields=%v", val))
		}
		if val, ok := args["view"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("view=%v", val))
		}
		// Handle multiple authentication parameters
		if cfg.BearerToken != "" {
			queryParams = append(queryParams, fmt.Sprintf("access_token=%s", cfg.BearerToken))
		}
		if cfg.APIKey != "" {
			queryParams = append(queryParams, fmt.Sprintf("key=%s", cfg.APIKey))
		}
		if cfg.BearerToken != "" {
			queryParams = append(queryParams, fmt.Sprintf("oauth_token=%s", cfg.BearerToken))
		}
		queryString := ""
		if len(queryParams) > 0 {
			queryString = "?" + strings.Join(queryParams, "&")
		}
		url := fmt.Sprintf("%s/projects/%s/datasets/%s/tables/%s%s", cfg.BaseURL, projectId, datasetId, tableId, queryString)
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Failed to create request", err), nil
		}
		// Set authentication based on auth type
		// Handle multiple authentication parameters
		// API key already added to query string
		// API key already added to query string
		// API key already added to query string
		req.Header.Set("Accept", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Request failed", err), nil
		}
		defer resp.Body.Close()

		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Failed to read response body", err), nil
		}

		if resp.StatusCode >= 400 {
			return mcp.NewToolResultError(fmt.Sprintf("API error: %s", body)), nil
		}
		// Use properly typed response
		var result models.Table
		if err := json.Unmarshal(body, &result); err != nil {
			// Fallback to raw text if unmarshaling fails
			return mcp.NewToolResultText(string(body)), nil
		}

		prettyJSON, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			return mcp.NewToolResultErrorFromErr("Failed to format JSON", err), nil
		}

		return mcp.NewToolResultText(string(prettyJSON)), nil
	}
}

func CreateBigquery_tables_getTool(cfg *config.APIConfig) models.Tool {
	tool := mcp.NewTool("get_projects_projectId_datasets_datasetId_tables_tableId",
		mcp.WithDescription("Gets the specified table resource by table ID. This method does not return the data in the table, it only returns the table resource, which describes the structure of this table."),
		mcp.WithString("projectId", mcp.Required(), mcp.Description("Required. Project ID of the requested table")),
		mcp.WithString("datasetId", mcp.Required(), mcp.Description("Required. Dataset ID of the requested table")),
		mcp.WithString("tableId", mcp.Required(), mcp.Description("Required. Table ID of the requested table")),
		mcp.WithString("selectedFields", mcp.Description("List of table schema fields to return (comma-separated). If unspecified, all fields are returned. A fieldMask cannot be used here because the fields will automatically be converted from camelCase to snake_case and the conversion will fail if there are underscores. Since these are fields in BigQuery table schemas, underscores are allowed.")),
		mcp.WithString("view", mcp.Description("Optional. Specifies the view that determines which table information is returned. By default, basic table information and storage statistics (STORAGE_STATS) are returned.")),
	)

	return models.Tool{
		Definition: tool,
		Handler:    Bigquery_tables_getHandler(cfg),
	}
}
