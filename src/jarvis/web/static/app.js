/**
 * Jarvis Posture Copilot — Client-side UX & HTMX Non-Interrupt Controller
 *
 * Prevents periodic HTMX background polling from interrupting:
 * 1. Text input in active form fields (typing, focus, or uncommitted drafts).
 * 2. Text selection and clipboard copying (mouse dragging, active selection ranges, or copy actions).
 *
 * Preserves normal user-initiated actions (button clicks, form submissions, explicit UI events).
 */

(function () {
  'use strict';

  // Track whether primary mouse button is currently held down (user dragging/selecting)
  let isMouseDown = false;

  window.addEventListener(
    'mousedown',
    function (evt) {
      if (evt.button === 0) {
        isMouseDown = true;
      }
    },
    true
  );

  window.addEventListener(
    'mouseup',
    function () {
      isMouseDown = false;
    },
    true
  );

  /**
   * Determine whether the given target element or any element within it
   * intersects an active text selection in the window.
   */
  function hasSelectionIntersecting(target) {
    if (!target) return false;
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed && sel.rangeCount > 0 && sel.toString().trim().length > 0) {
      for (let i = 0; i < sel.rangeCount; i++) {
        const range = sel.getRangeAt(i);
        try {
          if (range.intersectsNode && range.intersectsNode(target)) {
            return true;
          }
        } catch {
          if (
            target.contains(range.commonAncestorContainer) ||
            range.commonAncestorContainer.contains(target)
          ) {
            return true;
          }
        }
      }
    }

    // Also check text selection inside inputs or textareas within target
    const active = document.activeElement;
    if (
      active &&
      target.contains(active) &&
      (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')
    ) {
      if (typeof active.selectionStart === 'number' && typeof active.selectionEnd === 'number') {
        if (active.selectionStart !== active.selectionEnd) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Determine whether any element in the document has an active text selection.
   */
  function hasAnySelection() {
    const sel = window.getSelection();
    if (sel && !sel.isCollapsed && sel.rangeCount > 0 && sel.toString().trim().length > 0) {
      return true;
    }
    const active = document.activeElement;
    if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {
      if (typeof active.selectionStart === 'number' && typeof active.selectionEnd === 'number') {
        if (active.selectionStart !== active.selectionEnd) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Check whether the target element contains an actively focused text input,
   * textarea, or editable element, or contains any text field with unsaved content.
   */
  function hasActiveInput(target) {
    if (!target) return false;

    // Check currently focused element
    const active = document.activeElement;
    if (active && target.contains(active)) {
      const tag = active.tagName;
      if (tag === 'TEXTAREA' || active.isContentEditable) {
        return true;
      }
      if (tag === 'INPUT') {
        const type = (active.type || '').toLowerCase();
        if (!['button', 'submit', 'reset', 'checkbox', 'radio', 'hidden'].includes(type)) {
          return true;
        }
      }
    }

    // Check if any text input inside target has non-empty unsubmitted text
    const textInputs = target.querySelectorAll(
      'input:not([type="hidden"]):not([type="checkbox"]):not([type="radio"]):not([type="submit"]):not([type="button"]), textarea'
    );
    for (let i = 0; i < textInputs.length; i++) {
      if (textInputs[i].value && textInputs[i].value.trim().length > 0) {
        return true;
      }
    }

    return false;
  }

  /**
   * Check whether an HTMX request was triggered by periodic polling (hx-trigger="every ...")
   * vs. an explicit user interaction (click, submit, custom event).
   */
  function isPollingRequest(evt) {
    const reqConfig = evt.detail && evt.detail.requestConfig;
    const triggeringEvent = reqConfig && reqConfig.triggeringEvent;

    // In HTMX, polling triggers (hx-trigger="every Xs") do not pass a triggering DOM event.
    if (!triggeringEvent) {
      return true;
    }
    if (triggeringEvent.type === 'hx:poll:trigger') {
      return true;
    }
    return false;
  }

  /**
   * Check whether a polling update on the given target should be blocked
   * to avoid interrupting user typing or text copying.
   */
  function shouldBlockUpdate(target, evt) {
    if (!isPollingRequest(evt)) {
      // User clicks, form submits, and custom events proceed immediately
      return false;
    }

    // 1. Mouse is held down (user is dragging to select text or interacting)
    if (isMouseDown) {
      return true;
    }

    // 2. Active text selection intersecting this target or anywhere on page
    if (hasSelectionIntersecting(target) || hasAnySelection()) {
      return true;
    }

    // 3. User is actively typing or has unsaved text in an input inside target
    if (hasActiveInput(target)) {
      return true;
    }

    return false;
  }

  // Intercept before HTMX sends a network request
  document.addEventListener('htmx:beforeRequest', function (evt) {
    const target = evt.detail.target || evt.detail.elt;
    if (shouldBlockUpdate(target, evt)) {
      evt.preventDefault();
    }
  });

  // Intercept before HTMX swaps new HTML into the DOM
  document.addEventListener('htmx:beforeSwap', function (evt) {
    const target = evt.detail.target || evt.detail.elt;
    if (shouldBlockUpdate(target, evt)) {
      evt.preventDefault();
      if (evt.detail) {
        evt.detail.shouldSwap = false;
      }
    }
  });

  // Clean form input reset on successful assistant response
  document.addEventListener('htmx:afterOnLoad', function (evt) {
    if (evt.detail && evt.detail.elt && evt.detail.elt.id === 'ask-form') {
      const input = document.getElementById('ask-question-input');
      if (input && evt.detail.xhr && evt.detail.xhr.status === 200) {
        input.value = '';
      }
    }
  });
})();
