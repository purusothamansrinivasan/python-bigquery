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

func Bigquery_tabledata_listHandler(cfg *config.APIConfig) func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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
		if val, ok := args["formatOptions.useInt64Timestamp"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("formatOptions.useInt64Timestamp=%v", val))
		}
		if val, ok := args["maxResults"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("maxResults=%v", val))
		}
		if val, ok := args["pageToken"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("pageToken=%v", val))
		}
		if val, ok := args["selectedFields"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("selectedFields=%v", val))
		}
		if val, ok := args["startIndex"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("startIndex=%v", val))
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
		url := fmt.Sprintf("%s/projects/%s/datasets/%s/tables/%s/data%s", cfg.BaseURL, projectId, datasetId, tableId, queryString)
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
		var result models.TableDataList
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

func CreateBigquery_tabledata_listTool(cfg *config.APIConfig) models.Tool {
	tool := mcp.NewTool("get_projects_projectId_datasets_datasetId_tables_tableId_data",
		mcp.WithDescription("List the content of a table in rows."),
		mcp.WithString("projectId", mcp.Required(), mcp.Description("Required. Project id of the table to list.")),
		mcp.WithString("datasetId", mcp.Required(), mcp.Description("Required. Dataset id of the table to list.")),
		mcp.WithString("tableId", mcp.Required(), mcp.Description("Required. Table id of the table to list.")),
		mcp.WithBoolean("formatOptions.useInt64Timestamp", mcp.Description("Optional. Output timestamp as usec int64. Default is false.")),
		mcp.WithNumber("maxResults", mcp.Description("Row limit of the table.")),
		mcp.WithString("pageToken", mcp.Description("To retrieve the next page of table data, set this field to the string provided in the pageToken field of the response body from your previous call to tabledata.list.")),
		mcp.WithString("selectedFields", mcp.Description("Subset of fields to return, supports select into sub fields. Example: selected_fields = \"a,e.d.f\";")),
		mcp.WithString("startIndex", mcp.Description("Start row index of the table.")),
	)

	return models.Tool{
		Definition: tool,
		Handler:    Bigquery_tabledata_listHandler(cfg),
	}
}
