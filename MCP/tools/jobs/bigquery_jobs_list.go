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

func Bigquery_jobs_listHandler(cfg *config.APIConfig) func(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
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
		queryParams := make([]string, 0)
		if val, ok := args["allUsers"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("allUsers=%v", val))
		}
		if val, ok := args["maxCreationTime"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("maxCreationTime=%v", val))
		}
		if val, ok := args["maxResults"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("maxResults=%v", val))
		}
		if val, ok := args["minCreationTime"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("minCreationTime=%v", val))
		}
		if val, ok := args["pageToken"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("pageToken=%v", val))
		}
		if val, ok := args["parentJobId"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("parentJobId=%v", val))
		}
		if val, ok := args["projection"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("projection=%v", val))
		}
		if val, ok := args["stateFilter"]; ok {
			queryParams = append(queryParams, fmt.Sprintf("stateFilter=%v", val))
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
		url := fmt.Sprintf("%s/projects/%s/jobs%s", cfg.BaseURL, projectId, queryString)
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
		var result models.JobList
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

func CreateBigquery_jobs_listTool(cfg *config.APIConfig) models.Tool {
	tool := mcp.NewTool("get_projects_projectId_jobs",
		mcp.WithDescription("Lists all jobs that you started in the specified project. Job information is available for a six month period after creation. The job list is sorted in reverse chronological order, by job creation time. Requires the Can View project role, or the Is Owner project role if you set the allUsers property."),
		mcp.WithString("projectId", mcp.Required(), mcp.Description("Project ID of the jobs to list.")),
		mcp.WithBoolean("allUsers", mcp.Description("Whether to display jobs owned by all users in the project. Default False.")),
		mcp.WithString("maxCreationTime", mcp.Description("Max value for job creation time, in milliseconds since the POSIX epoch. If set, only jobs created before or at this timestamp are returned.")),
		mcp.WithNumber("maxResults", mcp.Description("The maximum number of results to return in a single response page. Leverage the page tokens to iterate through the entire collection.")),
		mcp.WithString("minCreationTime", mcp.Description("Min value for job creation time, in milliseconds since the POSIX epoch. If set, only jobs created after or at this timestamp are returned.")),
		mcp.WithString("pageToken", mcp.Description("Page token, returned by a previous call, to request the next page of results.")),
		mcp.WithString("parentJobId", mcp.Description("If set, show only child jobs of the specified parent. Otherwise, show all top-level jobs.")),
		mcp.WithString("projection", mcp.Description("Restrict information returned to a set of selected fields")),
		mcp.WithArray("stateFilter", mcp.Description("Filter for job state")),
	)

	return models.Tool{
		Definition: tool,
		Handler:    Bigquery_jobs_listHandler(cfg),
	}
}
