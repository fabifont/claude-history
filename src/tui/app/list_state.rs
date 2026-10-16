use super::App;
use crate::history::Conversation;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

impl App {
    pub(super) fn filter_indices<I>(&self, indices: I) -> Vec<usize>
    where
        I: IntoIterator<Item = usize>,
    {
        filter_conversation_indices(
            indices,
            &self.conversations,
            &self.excluded_projects,
            self.workspace_filter,
            self.current_project_dir_name.as_deref(),
        )
    }

    pub(super) fn apply_filtered(&mut self, filtered: Vec<usize>) {
        self.filtered = filtered;
        self.selected = self.selection_anchor.take().and_then(|path| {
            self.filtered
                .iter()
                .position(|&index| self.conversations[index].path == path)
        });
        if self.selected.is_none() && !self.filtered.is_empty() {
            self.selected = Some(0);
        }
    }

    pub(super) fn clear_selection_anchor(&mut self) {
        self.selection_anchor = None;
    }

    pub(super) fn select_prev(&mut self) {
        self.clear_selection_anchor();
        if let Some(selected) = self.selected
            && selected > 0
        {
            self.selected = Some(selected - 1);
        }
    }

    pub(super) fn select_next(&mut self) {
        self.clear_selection_anchor();
        if let Some(selected) = self.selected
            && selected + 1 < self.filtered.len()
        {
            self.selected = Some(selected + 1);
        }
    }

    pub(super) fn select_first(&mut self) {
        self.clear_selection_anchor();
        if !self.filtered.is_empty() {
            self.selected = Some(0);
        }
    }

    pub(super) fn select_last(&mut self) {
        self.clear_selection_anchor();
        if !self.filtered.is_empty() {
            self.selected = Some(self.filtered.len() - 1);
        }
    }

    pub(super) fn select_page_up(&mut self) {
        self.clear_selection_anchor();
        if let Some(selected) = self.selected {
            self.selected = Some(selected.saturating_sub(10));
        }
    }

    pub(super) fn select_page_down(&mut self) {
        self.clear_selection_anchor();
        if let Some(selected) = self.selected {
            let new_selected = (selected + 10).min(self.filtered.len().saturating_sub(1));
            self.selected = Some(new_selected);
        }
    }

    pub(super) fn select_half_page_down(&mut self, viewport_height: usize) {
        self.clear_selection_anchor();
        if let Some(selected) = self.selected {
            let half_page = viewport_height / 2;
            let new_selected = (selected + half_page).min(self.filtered.len().saturating_sub(1));
            self.selected = Some(new_selected);
        }
    }

    pub(super) fn scroll_list(&mut self, delta: isize) {
        self.clear_selection_anchor();
        let Some(selected) = self.selected else {
            return;
        };

        let max = self.filtered.len().saturating_sub(1);
        let new_selected = if delta >= 0 {
            selected.saturating_add(delta as usize).min(max)
        } else {
            selected.saturating_sub((-delta) as usize)
        };
        self.selected = Some(new_selected);
    }

    pub(super) fn get_selected_path(&self) -> Option<PathBuf> {
        self.selected
            .and_then(|sel| self.filtered.get(sel))
            .map(|&idx| self.conversations[idx].path.clone())
    }

    pub(crate) fn get_selected_source(&self) -> Option<crate::history::Source> {
        self.get_selected_conversation_index()
            .map(|index| self.conversations[index].source)
    }

    pub(crate) fn has_multiple_sources(&self) -> bool {
        if let Some(snapshot) = &self.snapshot {
            return snapshot.has_multiple_sources();
        }
        let sources = self
            .conversations
            .iter()
            .map(|conversation| conversation.source)
            .collect::<HashSet<_>>();
        sources.len() > 1
    }

    pub(super) fn get_selected_conversation_index(&self) -> Option<usize> {
        self.selected
            .and_then(|sel| self.filtered.get(sel))
            .copied()
    }

    pub(crate) fn remove_selected_from_list(&mut self) {
        let Some(selected) = self.selected else {
            return;
        };
        let Some(&conv_idx) = self.filtered.get(selected) else {
            return;
        };

        if let Some(snapshot) = &self.snapshot {
            // Rows stay put so row indices remain valid; the row is hidden
            // until a generation without it is shown.
            let doc = self
                .row_docs
                .get(conv_idx)
                .copied()
                .unwrap_or(super::NO_DOC);
            if doc != super::NO_DOC {
                self.removed.insert(snapshot.key_hash(doc));
            }
            self.filtered.retain(|&idx| idx != conv_idx);
            if self.filtered.is_empty() {
                self.selected = None;
            } else if selected >= self.filtered.len() {
                self.selected = Some(self.filtered.len() - 1);
            }
            self.send_index_scope();
            self.semantic_sent_scope_signature = None;
            self.request_reconcile();
            return;
        }

        Arc::make_mut(&mut self.conversations).remove(conv_idx);

        self.filtered.retain(|&idx| idx != conv_idx);
        for idx in &mut self.filtered {
            if *idx > conv_idx {
                *idx -= 1;
            }
        }

        if self.filtered.is_empty() {
            self.selected = None;
        } else if selected >= self.filtered.len() {
            self.selected = Some(self.filtered.len() - 1);
        }

        self.refresh_search_data();
    }
}

pub(super) fn filter_conversation_indices<I>(
    indices: I,
    conversations: &[Conversation],
    excluded_projects: &HashSet<String>,
    workspace_filter: bool,
    current_project_dir_name: Option<&str>,
) -> Vec<usize>
where
    I: IntoIterator<Item = usize>,
{
    indices
        .into_iter()
        .filter(|&idx| {
            conversations[idx]
                .project_name
                .as_deref()
                .is_none_or(|name| !project_is_excluded(name, excluded_projects))
        })
        .filter(|&idx| {
            let Some(project_dir_name) = current_project_dir_name.filter(|_| workspace_filter)
            else {
                return true;
            };
            if conversations[idx].source != crate::history::Source::Claude {
                let Ok(current) = std::env::current_dir() else {
                    return false;
                };
                let current = current.canonicalize().unwrap_or(current);
                return conversations[idx]
                    .project_path
                    .as_ref()
                    .or(conversations[idx].cwd.as_ref())
                    .is_some_and(|path| {
                        path.canonicalize().unwrap_or_else(|_| path.clone()) == current
                    });
            }
            conversations[idx]
                .path
                .parent()
                .and_then(|p| p.file_name())
                .is_some_and(|name| {
                    crate::history::path::is_same_project(&name.to_string_lossy(), project_dir_name)
                })
        })
        .collect()
}

/// The workspace a card must belong to, resolved once per card walk.
pub(super) struct WorkspaceScope<'a> {
    project_dir_name: Option<&'a str>,
    current_dir: Option<String>,
}

impl<'a> WorkspaceScope<'a> {
    /// `project_dir_name` is `None` when the workspace filter is off.
    pub(super) fn new(project_dir_name: Option<&'a str>) -> Self {
        let current_dir = project_dir_name.and_then(|_| {
            let current = std::env::current_dir().ok()?;
            let current = current.canonicalize().unwrap_or(current);
            Some(current.to_string_lossy().into_owned())
        });
        Self {
            project_dir_name,
            current_dir,
        }
    }
}

/// Card counterpart of `filter_conversation_indices`, decided per project.
pub(super) fn card_project_visible(
    project: &crate::store::ProjectView,
    excluded_projects: &HashSet<String>,
    workspace: &WorkspaceScope,
) -> bool {
    if project
        .name
        .is_some_and(|name| project_is_excluded(name, excluded_projects))
    {
        return false;
    }
    let Some(project_dir_name) = workspace.project_dir_name else {
        return true;
    };
    if project.key_is_path {
        workspace.current_dir.as_deref() == Some(project.workspace_key)
    } else {
        crate::history::path::is_same_project(project.workspace_key, project_dir_name)
    }
}

fn project_is_excluded(project_name: &str, excluded_projects: &HashSet<String>) -> bool {
    excluded_projects.contains(project_name)
        || project_name
            .split_once('/')
            .is_some_and(|(parent, _)| excluded_projects.contains(parent))
}
