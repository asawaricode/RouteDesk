import "dotenv/config";
import { supabase } from "./src/supabase.js";

const BASE_URL = process.env.API_BASE_URL || "http://127.0.0.1:4000/api/tickets";

async function runFeedbackTestSuite() {
  console.log("============================================================");
  console.log("RUNNING BACKEND FEEDBACK ENDPOINT TEST SUITE");
  console.log("============================================================");

  // 1. Fetch tickets to get an existing ticket ID
  const ticketsRes = await fetch(BASE_URL);
  if (!ticketsRes.ok) {
    throw new Error(`Failed to fetch tickets from ${BASE_URL}: HTTP ${ticketsRes.status}`);
  }
  const tickets = await ticketsRes.json();
  if (!tickets || tickets.length === 0) {
    throw new Error("No existing tickets in database to test feedback with.");
  }

  const testTicket = tickets[tickets.length - 1];
  console.log(`[*] Target test ticket: ID #${testTicket.id} (Predicted: '${testTicket.predicted_queue}')`);

  // Reset any prior feedback for this test ticket
  await supabase.from("ticket_feedback").delete().eq("ticket_id", testTicket.id);

  // Test 1: Invalid Ticket ID (400)
  const resBadId = await fetch(`${BASE_URL}/invalid-id/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_correct: true }),
  });
  console.log(`[+] Test 1 (Invalid Ticket ID): HTTP ${resBadId.status} (expected 400)`);
  if (resBadId.status !== 400) throw new Error(`Test 1 failed with status ${resBadId.status}`);

  // Test 2: Non-existent Ticket ID (404)
  const resNotFound = await fetch(`${BASE_URL}/999999999/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_correct: true }),
  });
  console.log(`[+] Test 2 (Non-existent Ticket): HTTP ${resNotFound.status} (expected 404)`);
  if (resNotFound.status !== 404) throw new Error(`Test 2 failed with status ${resNotFound.status}`);

  // Test 3: Invalid is_correct type (400)
  const resBadType = await fetch(`${BASE_URL}/${testTicket.id}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_correct: "yes" }),
  });
  console.log(`[+] Test 3 (Non-boolean is_correct): HTTP ${resBadType.status} (expected 400)`);
  if (resBadType.status !== 400) throw new Error(`Test 3 failed with status ${resBadType.status}`);

  // Test 4: is_correct: false without corrected_queue (400)
  const resMissingQueue = await fetch(`${BASE_URL}/${testTicket.id}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_correct: false }),
  });
  console.log(`[+] Test 4 (Missing corrected_queue on false): HTTP ${resMissingQueue.status} (expected 400)`);
  if (resMissingQueue.status !== 400) throw new Error(`Test 4 failed with status ${resMissingQueue.status}`);

  // Test 5: is_correct: false with unrecognized department (400)
  const resInvalidQueue = await fetch(`${BASE_URL}/${testTicket.id}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_correct: false, corrected_queue: "NonExistentDept" }),
  });
  console.log(`[+] Test 5 (Invalid queue category): HTTP ${resInvalidQueue.status} (expected 400)`);
  if (resInvalidQueue.status !== 400) throw new Error(`Test 5 failed with status ${resInvalidQueue.status}`);

  // Test 6: Successful feedback submission (is_correct: false, valid queue) (201)
  const targetReassignQueue = testTicket.predicted_queue === "IT Support" ? "Customer Service" : "IT Support";
  const resSuccess = await fetch(`${BASE_URL}/${testTicket.id}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_correct: false, corrected_queue: targetReassignQueue }),
  });
  const successData = await resSuccess.json();
  console.log(`[+] Test 6 (Successful submission): HTTP ${resSuccess.status} (expected 201), reassigned to: '${successData.feedback?.corrected_queue}'`);
  if (resSuccess.status !== 201) throw new Error(`Test 6 failed with status ${resSuccess.status}`);

  // Test 7: Prevent duplicate feedback submission for the same ticket (409)
  const resDuplicate = await fetch(`${BASE_URL}/${testTicket.id}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ is_correct: true }),
  });
  const dupData = await resDuplicate.json();
  console.log(`[+] Test 7 (Duplicate prevention): HTTP ${resDuplicate.status} (expected 409), error: '${dupData.error}'`);
  if (resDuplicate.status !== 409) throw new Error(`Test 7 failed with status ${resDuplicate.status}`);

  // Test 8: GET /api/tickets returns the attached feedback object
  const ticketsAfter = await (await fetch(BASE_URL)).json();
  const matched = ticketsAfter.find((t) => t.id === testTicket.id);
  const feedbackAttached = !!matched?.ticket_feedback;
  console.log(`[+] Test 8 (GET /api/tickets returns feedback): ${feedbackAttached ? "PASS" : "FAIL"}`);
  if (!feedbackAttached) throw new Error("Test 8 failed: ticket_feedback not attached to ticket");

  // Clean up test feedback record
  await supabase.from("ticket_feedback").delete().eq("ticket_id", testTicket.id);
  console.log("[OK] Test feedback cleaned up.");
  console.log("============================================================");
  console.log("ALL FEEDBACK TESTS PASSED SUCCESSFULLY!");
  console.log("============================================================");
}

runFeedbackTestSuite()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("[FAIL]", err.message);
    process.exit(1);
  });
