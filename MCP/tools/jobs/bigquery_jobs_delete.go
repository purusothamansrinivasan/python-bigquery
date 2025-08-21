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

func Bigquery_jobs_deleteHandler(cfg *config.APIConfig) func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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
		jobIdVal, ok := args["jobId"]
		if !ok {
			return mcp.NewToolResultError("Missing required path parameter: jobId"), nil
		}
		jobId, ok := jobIdVal.(string)
		if !ok {
			return mcp.NewToolResultError("Invalid path parameter: jobId"), nil
		}
		queryParams := make([]string, 0)
		if val, ok := args["location"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("location=%v", val))
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
		url := fmt.Sprintf("%s/projects/%s/jobs/%s/delete%s", cfg.BaseURL, projectId, jobId, queryString)
		req, err := http.NewRequest("DELETE", url, nil)
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
		var result map[string]interface{}
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

func CreateBigquery_jobs_deleteTool(cfg *config.APIConfig) models.Tool {
	tool := mcp.NewTool("delete_projects_projectId_jobs_jobId_delete",
		mcp.WithDescription("Requests the deletion of the metadata of a job. This call returns when the job's metadata is deleted."),
		mcp.WithString("projectId", mcp.Required(), mcp.Description("Required. Project ID of the job for which metadata is to be deleted.")),
		mcp.WithString("jobId", mcp.Required(), mcp.Description("Required. Job ID of the job for which metadata is to be deleted. If this is a parent job which has child jobs, the metadata from all child jobs will be deleted as well. Direct deletion of the metadata of child jobs is not allowed.")),
		mcp.WithString("location", mcp.Description("The geographic location of the job. Required. See details at: https://cloud.google.com/bigquery/docs/locations#specifying_your_location.")),
	)

	return models.Tool{
		Definition: tool,
		Handler:    Bigquery_jobs_deleteHandler(cfg),
	}
}
